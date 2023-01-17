import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.preprocessing import LabelEncoder
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if dirname is not "uncover":
            print(os.path.join(dirname, filename))

time_series_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
time_series_train.Date = time_series_train.Date.apply(pd.to_datetime)
original_country = {country:time_series_train[time_series_train["Country_Region"]==country] for country in time_series_train["Country_Region"].unique()}
gov_measures = pd.read_csv("/kaggle/input/uncover/HDE/acaps-covid-19-government-measures-dataset.csv")
gov_measures.date_implemented = gov_measures.date_implemented.apply(pd.to_datetime)
country_specific = {country:gov_measures[gov_measures["country"]==country] for country in gov_measures["country"].unique()}
continents_map = {}
for ind,row in gov_measures.iterrows():
    continents_map[row["country"]] = row["region"]
# additional preparations
add_prep_list = ["US", "Australia", "France"]
for country in add_prep_list:
    country_sum = time_series_train.groupby(["Country_Region","Date"]).sum().reset_index()
    original_country[country] = country_sum[country_sum["Country_Region"]== country]
    original_country[country]
country_mapping = {"Korea Republic of":"Korea, South"}
country_specific_buffer = {}
for key in country_specific:
    if key in country_mapping:
        mapping = country_specific[key]
        country_specific_buffer[country_mapping[key]]= mapping
country_specific.update(country_specific_buffer)
for key in country_mapping:
    continents_map[country_mapping[key]] = continents_map[key]
country_specific["Sweden"].sort_values("date_implemented")
# mathematical functions ------------------------------------------
def exp_func(x, a, b, c,d):
 return a * np.exp(b * (x-c)) + d

def lin(x,a,b):
    return a*x+b

# data processing functions --------------------------------------

def fit_data(data,ind=0,step=10,func=None):

    if func is None:
        def lin(x,a,b):
            return a*x+b
        func=lin
    x = np.array(data["ConfirmedCases"].index)[max(0,int(ind-step/2)):int(ind+step/2)]
    y_log = np.log(np.array(data["ConfirmedCases"])[max(0,int(ind-step/2)):int(ind+step/2)])
    opt_lin, pcov_lin = curve_fit(func, x, y_log)

    return opt_lin, pcov_lin

def scan_data_list(data,day_list = [], step=10, all_slopes=False, lower_limit = 500):
    try:
        dates_order = np.array(data.Date)
        data=data.reset_index()
        data_fit = data[data["ConfirmedCases"] > lower_limit]
        first_ent = data_fit.index[0]
        number_entries = len(data_fit)
        slopes = []
        offsets = []
        dates_from = []
        dates_to = []
        changes = []
        first=0
    except IndexError:
        #print("Error with reading data")
        return None
    for i in day_list:
        i = i-first_ent
        if number_entries-i >= 3 and i >= 0 :
            params, quality = fit_data(data_fit,ind=i,step=10)
            params_1, quality_1 = fit_data(data_fit,ind=i+step,step=10)
            slopes.extend([params[0],params_1[0]])
            offsets.extend([params[1],params_1[1]])
            dates_from.append(dates_order[first_ent+i])
            changes.append(params[0]-params_1[0])
            dates_to.append(dates_order[min(first_ent+i+step,len(dates_order)-1)])
    if len(slopes) > 0:
        percent_changes = np.array(changes)/max(slopes)
        day_of_change_index =[round(linalg.solve(slopes[i]-slopes[i+1], offsets[i+1]-offsets[i])[0]) for i in range(0,len(slopes)-1,2)]
        day_of_change = [dates_order[min(max(int(day),0),len(dates_order)-1)] for day in day_of_change_index]
        df_result = pd.DataFrame({"date_switch":day_of_change, "date_from":dates_from,"date_to":dates_to, "growth_change":percent_changes})
        if all_slopes:
            return_all = [((slopes[i],slopes[i+1]), (offsets[i],offsets[i+1])) for i in range(0,len(slopes)-1,2)]
            return return_all
        return df_result
    else:
        return None
        

# Evaluate measures against reported cases
def evaluate_data(original_country,country_specific, gov_measures, delays=[13], min_cases=50, region_specific=None): 
    measure_df_final = pd.DataFrame(columns=gov_measures.measure.unique())
    measure_df = pd.DataFrame(columns=gov_measures.measure.unique())
    for delay in delays:

        for country in original_country:
            if country in country_specific and (region_specific is None or continents_map[country] == region_specific):
                measure_vector = {measure:[np.nan] for measure in gov_measures.measure.unique()}
                dates_order = np.array(original_country[country].Date)
                dates_map = {date:i for i,date in enumerate(original_country[country].Date)}
                day_list = []
                measure_list = []
                for index, row in country_specific[country].iterrows():
                    if row["date_implemented"] in dates_map:
                        day_list.append(dates_map[row["date_implemented"]])
                        measure_list.append(row["measure"])
                try:
                    #print(country)
                    #print("...")
                    day_list.sort()
                    day_list_set = list(set(day_list))
                    day_list_set.sort()
                    day_list_map = {day_list_set[j]:j for j in range(0, len(day_list_set))}
                    country_df = scan_data_list(original_country[country], day_list=day_list_set, step=delay, lower_limit=100)

                    for ind,day_df in country_df.iterrows():
                        measure_vector[measure_list[day_list_map[dates_map[day_df["date_from"]]]]].append(day_df["growth_change"])
                        for key in measure_vector:
                            measure_vector[key] = [np.nanmean(measure_vector[key])]
                        measure_df_local = pd.DataFrame(measure_vector)
                        measure_df = measure_df.append(measure_df_local)
                    #print("Done")
                except TypeError as e:
                    #print(f"Attribute Error: {e}")
                    pass
                except Exception as e:
                    #print(f"Something failed:{e}")
                    pass
    return measure_df
day_dict = {"France":[50], "US":[70],"Italy":[32],"Spain":[50],"Germany":[57], "Norway":[50],"Sweden":[50]}
lower_limit = 100
for j,country in enumerate(day_dict):
    country_df = scan_data_list(original_country[country], all_slopes=True, day_list=day_dict[country], step=13, lower_limit=lower_limit)
    leng_cases = len(original_country[country]["ConfirmedCases"])
    last_entry = original_country[country].reset_index()["ConfirmedCases"][leng_cases-1]
    last_slope = scan_data_list(original_country[country], all_slopes=True, day_list=[leng_cases-15], step=13, lower_limit=lower_limit)
    x = np.arange(0, leng_cases)
    y_1 = lin(x,country_df[0][0][0],country_df[0][1][0])
    y_2 = lin(x,country_df[0][0][1],country_df[0][1][1])
    print(f"Latest doubling time:{np.log(2)/last_slope[0][0][1]}")
    print(f"Latest number of cases: {last_entry}")
    original_country[country]["ConfirmedCases"].index -= original_country[country]["ConfirmedCases"].index[0]
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle(f"Fig.{j}: {country}")
    #axs[0].title("Test")
    np.log(original_country[country]["ConfirmedCases"]).plot(ax=axs[1])
    axs[1].plot(x,y_1)
    axs[1].plot(x,y_2)
    axs[1].set_ylim((0,17))
    axs[1].set_ylabel("log(C)")
    axs[1].set_xlabel(f"days from day with case {lower_limit}")
    axs[0].set_ylabel("log(C)")
    axs[0].set_xlabel(f"days from day with case {lower_limit}")
    np.log(original_country[country]["ConfirmedCases"]).plot(ax=axs[0])
    plt.show()

std_list = []
min_cases = 50
begin = 8
end = 22
region=None
for i in range(begin,end):
    measure_df = evaluate_data(original_country,country_specific, gov_measures,delays=[i], region_specific=region)
    #stds = (measure_df.std()[measure_df.count() > min_cases]/measure_df.mean()[measure_df.count() > min_cases]).sort_values()
    stds = (measure_df.std()[measure_df.count() > min_cases]).sort_values()
    std_list.append(stds.mean())
std_list
delay_x = np.arange(begin,end)
std_y = np.array(std_list)
plt.xlabel("days of delay")
plt.ylabel("mean of relative standard deviation across measures")
plt.plot(delay_x,std_y)
min_cases = 50
region = None
measure_df = evaluate_data(original_country,country_specific, gov_measures,delays=[12,13,14], min_cases=min_cases, region_specific=region)
len(measure_df)
measure_result_vec = measure_df.mean()[measure_df.count() > min_cases]
measure_result_vec.sort_values().plot(kind="barh", figsize=(7,7))
measure_result_vec
stds = (measure_df.std()[measure_df.count() > min_cases]/measure_df.mean()[measure_df.count() > min_cases]).sort_values(ascending=False)
stds.plot(kind="barh", figsize=(7,7))
print(f"Average erro: {stds.mean()}")
def pca_add(measure_df_new):
    measure_df_new = measure_df_new.dropna(axis=1, thresh=20)
    filler = [pd.Series(np.random.normal(item.mean(),item.std(),len(item))) for ind,item in measure_df_new.iteritems()]
    filler_frame = pd.DataFrame(filler).T
    filler_frame.columns = measure_df_new.columns
    measure_df_new = measure_df_new.reset_index().fillna(filler_frame).dropna(axis=1,how="all")
    measure_df_new = measure_df_new.drop(columns=["index"])
    pca = decomposition.PCA(n_components=projected_components)
    new = pca.fit_transform(measure_df_new)
    measure_vec = [key for key in measure_df_new]
    vec_len_list = []
    vec_list = []
    for number in range(0,projected_components):
        inds = (-1*pca.components_[number]).argsort()[:max_features]
        vec = pca.components_[number][inds]
        vec_length = np.sqrt(np.sum(vec*vec))
        vec_len_list.append(vec_length)
        vec_list.append(vec/vec_length)
    real_eigen_values = vec_len_list*pca.singular_values_
    inds_eigenvalues = (-1*real_eigen_values).argsort()
    return pca, measure_vec, vec_len_list, vec_list, real_eigen_values, inds_eigenvalues

def np_add(measure_df_new):
    corr = measure_df_new.corr()
    np.linalg.eig(corr.fillna(0))
    
    return
    
    
%%time
# Use PCA algorithm on cov matrix; we first filter for entries that have at least 2 entries 
# and then replaced NAN values with mean values. This lowers the importance of these measures in the PCA 
# analysis because the average error and the correltaion is thus lowerd. However, entries with only a few non-nan 
# entries do not carrie a lot of information and thus it may be ok to shift the results to the more frequently occuring
# strategies
# We normalize the resulting vectors from PCA and sort for the largest eigenvalues
# ----------------------------------------------------------
repeater_list = []
for repeater in range(0,1):
    min_cases = 50
    max_features = 38
    projected_components = 11
    measure_df_new = measure_df.copy()
    for key in measure_df_new:
        if not measure_df_new[key].count() > min_cases:
            measure_df_new.drop(columns=[key], inplace=True)
    measure_vec = [key for key in measure_df_new]
    pca, measure_vec, vec_len_list, vec_list, real_eigen_values, inds_eigenvalues = pca_add(measure_df_new)
    
    
    vec_series_list = []
    average_vec_series = {key:0 for key in measure_vec}
    for eig_inds in inds_eigenvalues:
        inds = (-1*pca.components_[eig_inds]).argsort()[:max_features]
        #print(vec_list[eig_inds])
        measure_vec_ind_list = []
        for ind in inds[:max_features]:
            measure_vec_ind_list.append(measure_vec[ind])
        vec_series = pd.Series(vec_list[eig_inds], index=measure_vec_ind_list)
        vec_series.plot(kind="barh")
        for vec_row,vec_val in vec_series.iteritems():
            average_vec_series[vec_row] += (1./real_eigen_values[eig_inds])*(vec_val if vec_val != np.nan else 0)
        vec_series_list.append((1./real_eigen_values[eig_inds])*vec_series)
        plt.show()
        print(f"Weight: {1/real_eigen_values[eig_inds]}")
        print("--------------")
    repeater_list.extend(vec_series_list)
    

#vec_series = None
#for vec_series_ent in vec_series_list:
#    print(vec_series_ent)
#    if vec_series is None:
#        vec_series = vec_series_ent
#    else:
#        vec_series += vec_series_ent
#print(vec_series)   
average_vec_series
repeater_list
add = None
for item in repeater_list:
    if add is None:
        add = item
    else:
        add += item
(add/len(repeater_list)).sort_values()
add = None
for item in repeater_list:
    if add is None:
        add = item
    else:
        add += item
(add/len(repeater_list)).sort_values()
min_cases = 50
corr = measure_df.corr()[measure_df.count()>min_cases].T[measure_df.count()>min_cases]
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()

final_matrix
ax = sns.heatmap(
    final_matrix, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
min_cases = 120
corr = measure_df.corr()[measure_df.count()>min_cases].T[measure_df.count()>min_cases]
measure_result_vec = measure_df.mean()[measure_df.count()>min_cases]
measure_vec = [key for key in measure_df.mean()[measure_df.count()>min_cases].index]
data = np.linalg.eig(corr.fillna(0))[1]
eig = np.linalg.eig(corr.fillna(0))[0]
zipped = list(zip(eig,data))
res = sorted(zipped, key = lambda x: x[0])
df = None
for dat in res:
    if df is None:
        df = pd.Series(dat[1], index=measure_vec)
    else:
        df += pd.Series(dat[1], index=measure_vec)
df = pd.Series(dat[1], index=measure_vec).sort_values() 
#df = df.where(df != 0 ).dropna()
df.plot(kind="barh")
plt.show()
new_add = 0
for dat in res:
    df = pd.Series(dat[1], index=measure_vec)
    new = df*measure_result_vec
    print("HEre ZEro:")
    print((df*pd.Series(res[1][1], index=measure_vec)).sum())
    new_add += new["Border checks"]
    #new.plot(kind="barh")
    #df = df.where(df != 0 ).dropna()
    df.sort_values().plot(kind="barh")
    print(dat[0])
    print(new.sum())
    plt.show()
print(new_add)
min_cases = 50
measure_df.corr()[measure_df.count()>min_cases].T[measure_df.count()>min_cases]
measure_vec = [key for key in measure_df.mean()[measure_df.count()>min_cases].index]
mean_matrix = np.diag(measure_df.mean()[measure_df.count()>min_cases])
mean_array = np.array(np.diagonal(mean_matrix))
div = np.array([[ent/abs(ent)] for ent in data[:,0]])
data_new = data/div

new_basis_mean = np.diag(np.diagonal((data.dot(mean_matrix).dot(data.T))))
new_basis_mean
backtransform_mean = np.diag(data.T.dot(new_basis_mean).dot(data))
backtransform_mean
pd.Series(backtransform_mean, index=measure_vec).sort_values()

mean_matrix_free_basis = np.diag((np.sum(mean_array*data_new,axis=1)))
final_matrix = (data_new.T @ mean_matrix_free_basis/(data_new@np.ones(15))@ data_new)

np.diagonal(final_matrix)
res = pd.Series(np.diagonal(final_matrix), index=measure_vec) #.sort_values()
res
final_matrix @ np.ones(15)
pd.Series(final_matrix @ np.ones(15), index=measure_vec).sort_values()
#np.ones(18)@final_matrix@np.ones(18

pd.Series(final_matrix @ np.array([0,0,0,1,0,1,1,0,0,1,1,0,1,0,0]), index=measure_vec).sort_values()
print({i:((final_matrix@data_new[i])/data_new[i]).mean() for i in range(len(data_new)-1)})
pd.Series(data_new[6], index=measure_vec).sort_values().plot(kind="barh")
res.sort_values()


from pandas import read_csv

data = read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
target = "Attrition"

feature_by_dtype = {}

for c in data.columns:

    if c == target: continue

    data_type = str(data[c].dtype)

    if data_type not in feature_by_dtype.keys():

        feature_by_dtype[data_type] = [c]

    else:

        feature_by_dtype[data_type].append(c)

        

print(feature_by_dtype)

feature_by_dtype.keys()
objects = feature_by_dtype["object"]
remove = ["Over18"]
categorical_features = [f for f in objects if f not in remove]
categorical_features
int64s = feature_by_dtype["int64"]
remove.append("StandardHours")

remove.append("EmployeeCount")
for i in int64s:

    print(i)

    #print(data[i])

    print(data[i].unique())
count_features = []

for i in [i for i in int64s if len(data[i].unique())<20 and i not in remove]:

    count_features.append(i)
count_features
count_features = count_features + ["TotalWorkingYears", "YearsAtCompany","HourlyRate"]

remove.append("EmployeeNumber")
numerical_features = [i for i in int64s if i not in remove]
numerical_features
data[numerial_features].head()
def display_ttest(data, category, numeric):

    output = {}

    s1 = data[data[category] == data[category].unique()[0]][numeric]

    s2 = data[data[category] == data[category].unique()[1]][numeric]

    from scipy.stats import ttest_ind

    t, p = ttest_ind(s1,s2)

    from IPython.display import display

    from pandas import DataFrame

    display(DataFrame(data=[{"t-test statistic" : t, "p-value" : p}], columns=["t-test statistic", "p-value"], index=[category]).round(2))



def display_ztest(data, category, numeric):

    output = {}

    s1 = data[data[category] == data[category].unique()[0]][numeric]

    s2 = data[data[category] == data[category].unique()[1]][numeric]

    from statsmodels.stats.weightstats import ztest

    z, p = ztest(s1,s2)

    from IPython.display import display

    from pandas import DataFrame

    display(DataFrame(data=[{"z-test statistic" : z, "p-value" : p}], columns=["z-test statistic", "p-value"], index=[category]).round(2))

    

def display_cxn_analysis(data, category, numeric, target):

    

    from seaborn import boxplot, kdeplot, set_style

    from matplotlib.pyplot import show, figure, subplots, ylabel, xlabel, subplot, suptitle

    

    not_target = [a for a in data[category].unique() if a != target][0]

    

    pal = {target : "yellow",

          not_target : "darkgrey"}

    



    set_style("whitegrid")

    figure(figsize=(12,5))

    suptitle(numeric + " by " + category)



    # ==============================================

    

    p1 = subplot(2,2,2)

    boxplot(y=category, x=numeric, data=data, orient="h", palette = pal)

    p1.get_xaxis().set_visible(False)



    # ==============================================

    

    p2 = subplot(2,2,4, sharex=p1)

    

    s1 = data[data[category] == target][numeric]

    s1 = s1.rename(target)

    kdeplot(s1, shade=True, color = pal[target])

    

    s2 = data[data[category] == not_target][numeric]

    s2 = s2.rename(not_target)  

    kdeplot(s2, shade=True, color = pal[not_target])

    

    ylabel("Density Function")

    xlabel(numeric)

    

    # ==============================================

    

    p3 = subplot(1,2,1)

    from seaborn import pointplot

    from matplotlib.pyplot import rc_context



    with rc_context({'lines.linewidth': 0.8}):

        pp = pointplot(x=category, y=numeric, data=data, capsize=.1, color="black", marker="s")

        

    

    # ==============================================

    

    show()

    

    #display p value

    

    if(data[category].value_counts()[0] > 30 and data[category].value_counts()[1] > 30):

        display_ztest(data,category,numeric)

    else:

        display_ttest(data,category,numeric)

    

    #Means, Standard Deviation, Absolute Distance

    table = data[[category,numeric]]

    

    means = table.groupby(category).mean()

    stds = table.groupby(category).std()

    

    s1_mean = means.loc[data[category].unique()[0]]

    s1_std = stds.loc[data[category].unique()[0]]

    

    s2_mean = means.loc[data[category].unique()[1]]

    s2_std = means.loc[data[category].unique()[1]]

    

    print("%s Mean: %.2f (+/- %.2f)" % (category + " == " + str(data[category].unique()[0]),s1_mean, s1_std))

    print("%s Mean : %.2f (+/- %.2f)" % (category + " == " + str(data[category].unique()[1]), s2_mean, s2_std))

    print("Absolute Mean Diferrence Distance: %.2f" % abs(s1_mean - s2_mean))
def get_p_value(s1,s2):

    

    from statsmodels.stats.weightstats import ztest

    from scipy.stats import ttest_ind

    

    if(len(s1) > 30 & len(s2) > 30):

        z, p = ztest(s1,s2)

        return p

    else:

        t, p = ttest_ind(s1,s2)

        return p

    

def get_p_values(data, category, numerics):

    

    output = {}

    

    for numeric in numerics:

        s1 = data[data[category] == data[category].unique()[0]][numeric]

        s2 = data[data[category] == data[category].unique()[1]][numeric]

        row = {"p-value" : get_p_value(s1,s2)}

        output[numeric] = row

    

    from pandas import DataFrame

    

    return DataFrame(data=output).T



def get_statistically_significant_numerics(data, category, numerics):

    df = get_p_values(data, category, numerics)

    return list(df[df["p-value"] < 0.05].index)



def get_statistically_non_significant_numerics(data, category, numerics):

    df = get_p_values(data, category, numerics)

    return list(df[df["p-value"] >= 0.05].index)

    

def display_p_values(data, category, numerics):

    from IPython.display import display

    display(get_p_values(data, category, numerics).round(2).sort_values("p-value", ascending=False))
get_p =  get_p_values(data,target,numerical_features)

a = get_p["p-value"]<0.05

a



##def get_p_values(data, target, numerical_features):

    

output = {}

    

for numeric in numerical_features:

    s1 = data[data[target] == data[target].unique()[0]][numeric]

    s2 = data[data[target] == data[target].unique()[1]][numeric]

    print("numeric")

    print(numeric)

    print("s1 : ")

    print(s1)

    print("s2 : ")

    print(s2)

        ##row = {"p-value" : get_p_value(s1,s2)}

        ##output[numeric] = row

    

    ##from pandas import DataFrame

    

    ##return DataFrame(data=output).T
significant = get_statistically_significant_numerics(data,target,numerical_features)

ns = get_statistically_non_significant_numerics(data,target,numerical_features)
significant
i = iter(significant)

i
display_cxn_analysis(data,target, next(i),"Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
display_cxn_analysis(data, target, next(i), "Yes")
for n in ns:

    print(n)

    

    display_cxn_analysis(data, target, n, "Yes")
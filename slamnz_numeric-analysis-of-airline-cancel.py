from pandas import read_csv

data = read_csv("../input/DelayedFlights.csv")
data = data.drop("Unnamed: 0",1)
target = ["Cancelled"]

leaky_features = ["Year", "Diverted", "ArrTime", "ActualElapsedTime", "AirTime", "CancellationCode", "ActualElapsedTime", "AirTime", "ArrDelay", "TaxiIn", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay","LateAircraftDelay"]

features = [x for x in data.columns if (x != target[0]) & (x not in leaky_features) & (len(data[x].unique().tolist()) > 1)]
where = data.Month.isin([10,11,12])

data = data[where]
def get_dtypes(data,features):

    output = {}

    for f in features:

        dtype = str(data[f].dtype)

        if dtype not in output.keys(): output[dtype] = [f]

        else: output[dtype] += [f]

    return output
dtypes = get_dtypes(data,features)
categories = ["Month", "DayOfWeek", "DayofMonth"]

categories += dtypes["object"]

numerics = [i for i in dtypes["int64"] if i not in categories]

numerics += dtypes["float64"]
data[numerics].head()
data[numerics].isnull().any()
from scipy.stats import pearsonr, spearmanr, kendalltau

from itertools import combinations



def get_correlation_table(data, numerical_features):

    

    output = []



    for n1, n2 in combinations(numerical_features, 2):



        data = data[(data[n1].notnull()) & (data[n2].notnull())]

        

        unit = {}



        unit["Numeric X"] = n1

        unit["Numeric Y"] = n2



        pearson = pearsonr(data[n1],data[n2])

        unit["Pearson"] = pearson[0]

        unit["Pearson p-value"] = pearson[1]



        spearman = spearmanr(data[n1],data[n2])

        unit["Spearman"] = spearman[0]

        unit["Spearman p-value"] = spearman[1]



        kendall = kendalltau(data[n1],data[n2])

        unit["Kendall"] = kendall[0]

        unit["Kendall p-value"] = kendall[1]



        output.append(unit)

    

    return output
from pandas import DataFrame

correlations = DataFrame(get_correlation_table(data,numerics))
from IPython.display import display

where = (correlations["Kendall"] < -0.3) | (correlations["Kendall"] > 0.30)

display(correlations[where].sort_values("Kendall", ascending=False).round(2))
from statsmodels.stats.weightstats import ztest

from scipy.stats import ttest_ind

from IPython.display import display

from pandas import DataFrame

from seaborn import boxplot, kdeplot, set_style, distplot

from matplotlib.pyplot import show, figure, subplots, ylabel, xlabel, subplot, suptitle



def display_ttest(data, category, numeric):

    output = {}

    s1 = data[data[category] == data[category].unique()[0]][numeric]

    s2 = data[data[category] == data[category].unique()[1]][numeric]

    t, p = ttest_ind(s1,s2)

    display(DataFrame(data=[{"t-test statistic" : t, "p-value" : p}], columns=["t-test statistic", "p-value"], index=[category]).round(2))



def display_ztest(data, category, numeric):

    output = {}

    s1 = data[data[category] == data[category].unique()[0]][numeric]

    s2 = data[data[category] == data[category].unique()[1]][numeric]

    z, p = ztest(s1,s2)

    display(DataFrame(data=[{"z-test statistic" : z, "p-value" : p}], columns=["z-test statistic", "p-value"], index=[category]).round(2))



def display_binary_cxn_analysis(data, category, numeric, target):

    

    data = data[data[numeric].notnull()]

    

    not_target = [a for a in data[category].unique() if a != target][0]

    

    pal = {target : "#b5615f",

          not_target : "#2c2a2f"}

    



    set_style("whitegrid")

    figure(figsize=(12,5))

    suptitle(numeric + " by " + category)



    # === === #

    

    p1 = subplot(2,2,2)

    boxplot(y=category, x=numeric, data=data, orient="h", palette = pal)

    p1.get_xaxis().set_visible(False)



    # === === #

    

    p2 = subplot(2,2,4, sharex=p1)

    

    s2 = data[data[category] == not_target][numeric]

    s2 = s2.rename(not_target)  

    distplot(s2, kde=False, color = pal[not_target])

    

    s1 = data[data[category] == target][numeric]

    s1 = s1.rename(target)

    distplot(s1, kde=False, color = pal[target])

    

    xlabel(numeric)

    

    # === ==== #

    

    p3 = subplot(1,2,1)

    from seaborn import pointplot

    from matplotlib.pyplot import rc_context



    with rc_context({'lines.linewidth': 0.8}):

        pp = pointplot(x=category, y=numeric, data=data, capsize=.1, color="black", marker="s")

        

    

    # === === #

    

    show()

    

    #display p value

    

    if(data[category].value_counts()[target] > 30 and data[category].value_counts()[not_target] > 30):

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
def get_significant_numeric_relationships(data,category,numerics):



    significant = []

    non_significant = []

    

    for numeric in numerics:



        data = (data.copy())[data[numeric].notnull()]



        value = iter(data[category].unique())



        sample_1 = data[data[category] == next(value)][numeric]

        sample_2 = data[data[category] == next(value)][numeric]



        z, p = ztest(sample_1,sample_2)



        if p < 0.05:



            significant += [numeric]



        else:



            non_significant += [numeric]



    return significant, non_significant
significant, non_significant = get_significant_numeric_relationships(data, target[0], numerics)
numeric = iter(significant)
display_binary_cxn_analysis(data, target[0], next(numeric), 1)
display_binary_cxn_analysis(data, target[0], next(numeric), 1)
display_binary_cxn_analysis(data, target[0], next(numeric), 1)
display_binary_cxn_analysis(data, target[0], next(numeric), 1)
display_binary_cxn_analysis(data, target[0], next(numeric), 1)
display_binary_cxn_analysis(data, target[0], next(numeric), 1)
non_significant
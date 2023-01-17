from pandas import read_csv

raw_data = read_csv("../input/Dataset_spine.csv")
raw_data.head()
# === Rename The Columns === #



column_names = ("pelvic_incidence",

"pelvic_tilt",

"lumbar_lordosis_angle",

"sacral_slope",

"pelvic_radius",

"degree_spondylolisthesis",

"pelvic_slope",

"Direct_tilt",

"thoracic_slope",

"cervical_tilt",

"sacrum_angle",

"scoliosis_slope")



# === Rename === #



rename = {}

for i in range(0,12):

    temp = "Col" + str(i+1)

    rename[temp] = column_names[i]



renamed_data = raw_data.rename(columns = rename)
data = renamed_data.drop("Unnamed: 13",1)

target = "Class_att"

features = [feature for feature in data.columns if feature != target]
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



from scipy.stats import ks_2samp

def display_ks_2samp(data,binary_category,numeric):

    func = ks_2samp

    output = {}

    s1 = data[data[binary_category] == data[binary_category].unique()[0]][numeric]

    s2 = data[data[binary_category] == data[binary_category].unique()[1]][numeric]

    from statsmodels.stats.weightstats import ztest

    R, p = func(s1,s2)

    from IPython.display import display

    from pandas import DataFrame

    display(DataFrame(data=[{func.__name__ : R, "p-value" : p}], columns=[func.__name__, "p-value"], index=[binary_category]).round(2))



from scipy.stats import ranksums

def display_ranksums(data,binary_category,numeric):

    func = ranksums

    output = {}

    s1 = data[data[binary_category] == data[binary_category].unique()[0]][numeric]

    s2 = data[data[binary_category] == data[binary_category].unique()[1]][numeric]

    from statsmodels.stats.weightstats import ztest

    R, p = func(s1,s2)

    from IPython.display import display

    from pandas import DataFrame

    display(DataFrame(data=[{func.__name__ : R, "p-value" : p}], columns=[func.__name__, "p-value"], index=[binary_category]).round(2))



    

def display_binary_cxn_analysis(data, category, numeric, target):

    

    from seaborn import boxplot, kdeplot, set_style

    from matplotlib.pyplot import show, figure, subplots, ylabel, xlabel, subplot, suptitle

    

    not_target = [a for a in data[category].unique() if a != target][0]

    

    pal = {target : "orange",

          not_target : "darkgrey"}

    



    set_style("whitegrid")

    figure(figsize=(12,5))

    suptitle(numeric + " by " + category)



    # === === #

    

    p1 = subplot(2,2,2)

    boxplot(y=category, x=numeric, data=data, orient="h", palette = pal)

    p1.get_xaxis().set_visible(False)



    # === === #

    

    p2 = subplot(2,2,4, sharex=p1)

    

    s1 = data[data[category] == target][numeric]

    s1 = s1.rename(target)

    kdeplot(s1, shade=True, color = pal[target])

    

    s2 = data[data[category] == not_target][numeric]

    s2 = s2.rename(not_target)  

    kdeplot(s2, shade=True, color = pal[not_target])

    

    ylabel("Density Function")

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

        

    display_ranksums(data,category,numeric)

    

    display_ks_2samp(data,category,numeric)

    

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
for feature in features:

    display_binary_cxn_analysis(data,target,feature,"Abnormal")
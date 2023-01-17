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
objects = feature_by_dtype["object"]
remove = ["Over18"]
categorical_features = [f for f in objects if f not in remove]
int64s = feature_by_dtype["int64"]
remove.append("StandardHours")

remove.append("EmployeeCount")
count_features = []

for i in [i for i in int64s if len(data[i].unique()) < 10 and i not in remove]:

    count_features.append(i)
#count_features = count_features + ["TotalWorkingYears", "YearsAtCompany", "HourlyRate"]
data[count_features].head()
data[categorical_features].head()
from scipy.stats import chi2_contingency

from pandas import crosstab, DataFrame



p_value_table = DataFrame(index = [target], columns = (categorical_features+count_features))



for c in (categorical_features+count_features):



    crosstable = crosstab(data[c], data[target])

    chi2, p, dof, expected = chi2_contingency(crosstable)

    p_value_table[c][target] = p



p_value_table = p_value_table.T

p_value_table["p < 0.05"] = p_value_table.apply(lambda x : x < 0.05)
p_value_table.sort_values("Attrition", ascending=False)
ns = p_value_table[p_value_table["p < 0.05"] == False].index.tolist()

print(ns)
significant = p_value_table[p_value_table["p < 0.05"] == True].index.tolist()

print(significant)
def percentages(data,category, filter_):

    output = {}

    total_count = data[filter_][category].value_counts().sum()

    for subclass in data[filter_][category].unique():

        subclass_count = data[filter_][category].value_counts()[subclass]

        output[subclass] = (subclass_count / total_count) * 100

    return output



from IPython.display import display

from pandas import DataFrame, options



def display_percentages(data,category, filter_):

    perc = percentages(data,category, filter_)

    df = DataFrame(perc, index=["Percent"]).T.sort_values("Percent", ascending=False)

    df["Cumulative Percent"] = [df["Percent"][0:i].sum() for i in range(1,len(df)+1)]

    options.display.float_format = '{:,.1f}%'.format

    print("Yes Only")

    print("Total Count: %s" % len(data[filter_]))

    display(df)

    

#====



from seaborn import countplot, despine, axes_style, set_style

from matplotlib.pyplot import show,figure,subplot,xticks,suptitle,title, ylabel, xlabel, margins

from numpy import mean



def display_categorical_x_categorical_analysis(data,category):



    set_style("whitegrid")



    with axes_style({'grid.color': "0.95", "lines.color" : "0.95"}):



        c = category



        order = data[data[target] == "Yes"][c].value_counts().sort_values(ascending=False).index



        fig = figure(figsize=(12,6))

        suptitle(c, fontsize=16)

        margins(0.8)

        subplot(121)

        title("Yes Only")

        cp = countplot(x=c, data=data[data[target] == "Yes"], order=order, color="#121831", linewidth=0)

        despine(left=True, top=True)



        xlabel_char_length = int(mean([len(str(i)) for i in data[c].unique()]))



        if(xlabel_char_length in range(7, 15)): 

            xticks(rotation=45)

        elif(xlabel_char_length > 14):

            xticks(rotation=90)



        subplot(122)

        title("Yes vs No")

        cp = countplot(x=c, hue=target, data=data, order=order, palette=["#121831", "#d4e2ed"], linewidth=0)

        despine(left=True, top=True)

        if(xlabel_char_length in range(7, 15)): 

            xticks(rotation=45)

        elif(xlabel_char_length > 14):

            xticks(rotation=90)

        xlabel(c)

        show()



        display_percentages(data,c,data[target] == "Yes")
i = iter(significant)
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
display_categorical_x_categorical_analysis(data,next(i))
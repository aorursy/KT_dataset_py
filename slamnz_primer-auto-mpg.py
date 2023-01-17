def get_dtype_lists(data):

    features = data.columns

    output = {}

    for f in features:

        dtype = str(data[f].dtype)

        if dtype not in output.keys(): output[dtype] = [f]

        else: output[dtype] += [f]

    return output



def show_uniques(data,features):

    for f in features:

        if len(data[f].unique()) < 30:

            print("%s: count(%s) %s" % (f,len(data[f].unique()),data[f].unique()))

        else:

            print("%s: count(%s) %s" % (f,len(data[f].unique()),data[f].unique()[0:10]))



def show_all_uniques(data,features):

    dtypes = get_dtype_lists(data)

    for key in dtypes.keys():

        print(key + "\n")

        show_uniques(data,dtypes[key])
from pandas import read_csv

data = read_csv("../input/auto-mpg.csv")
data.shape
show_all_uniques(data, data.columns)
data.isnull().any()
dtypes = get_dtype_lists(data)

numerics = dtypes["float64"] + ["weight"]

categories = dtypes["object"] + dtypes["int64"]

categories.remove("weight")
from seaborn import distplot, boxplot, countplot, set_style,despine, axes_style, set_palette, color_palette

from matplotlib.pyplot import subplot, show

from IPython.display import display

from pandas import DataFrame



# === Numeric Analysis === #



def numeric_analysis(series):

    

    no_nulls = series.dropna()

    

    with axes_style({"axes.grid": False}):

        

        cell_1 = subplot(211)

        dp = distplot(no_nulls, kde=True)

        dp.set_xlabel("",visible=False)

        dp.set_yticklabels(dp.get_yticklabels(),visible=False)

        despine(left = True)



        cell_2 = subplot(212, sharex=cell_1)

        boxplot(no_nulls)

        despine(left=True)

    

    show()

    

    display(DataFrame(series.describe().round(2)).T)
for numeric in numerics: numeric_analysis(data[numeric])
from seaborn import countplot, color_palette, set_style, despine

from pandas import DataFrame

from matplotlib.pyplot import show, figure

from IPython.display import display



set_style("whitegrid")

set_style({"axes.grid":False})



for c in categories:

    

    figure(figsize=(12.5,0.5*data[c].unique().size))

    

    if data[c].unique().size > 10:

  

        ax = countplot(data=data, y=c, palette=color_palette("colorblind"))

        

        last_tick = int(round(ax.get_xticks()[-1]/len(data),1) * 10) + 1

        ax.set_xticks([i * (len(data) * 0.1) for i in range(0,last_tick)])

        ax.set_xticklabels(["{:.0f}%".format((tick / len(data)) * 100) for tick in ax.get_xticks()])

        

        despine(left=True)

        show()

        display(DataFrame(data[c].value_counts()).T)

        continue

        

    ax = countplot(data[c], palette=color_palette("colorblind"))

    

    last_tick = int(round(ax.get_yticks()[-1]/len(data),1) * 10) + 1

    ax.set_yticks([i * (len(data) * 0.1) for i in range(0,last_tick)])

    ax.set_yticklabels(["{:.0f}%".format((tick / len(data)) * 100) for tick in ax.get_yticks()])

    

    maximum_yticklabel_length = max([len(str(x)) for x in data[c].unique()])

    

    if maximum_yticklabel_length in range (5,7):

        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

    elif maximum_yticklabel_length > 6:

        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

        

    despine(left=True)

    

    show()

    

    display(DataFrame(data[c].value_counts()).T)
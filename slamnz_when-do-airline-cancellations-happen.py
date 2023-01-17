from pandas import read_csv

data = read_csv("../input/DelayedFlights.csv")
days = {1:"Mon",

       2:"Tues",

       3:"Wed",

       4:"Thur",

       5:"Fri",

       6:"Sat",

       7:"Sun"}



data["DayOfWeek"] = data["DayOfWeek"].apply(lambda x: days[x])
data = data[data["Cancelled"] == 1]
from seaborn import countplot, set_style,despine, axes_style, set_palette, color_palette

from matplotlib.pyplot import subplot, show

from IPython.display import display

from pandas import DataFrame, set_option



set_option("display.max_columns",200)



# === Category Analysis === #

    

def category_analysis(series):

    

    set_style("whitegrid")

    

    with axes_style({'axes.grid': False}):

        cp = countplot(series, palette=color_palette("colorblind"))

        cp.set_title(cp.get_xlabel())

        cp.set_xlabel("",visible=False)

        despine()

    

    show()

    

    percent = series.value_counts().apply(lambda x: "{:.2f}%".format(x / len(series) * 100))

    percent.name = "Percent"

    count = series.value_counts()

    count.name = "Count"

    output = DataFrame([percent,count])

    display(output)
series = data["Month"]



set_style("whitegrid")

    

with axes_style({'axes.grid': False}):

    

    pal = {12: color_palette("colorblind")[4]}

    pal[10] = pal[11] = color_palette("colorblind")[0]

    

    cp = countplot(series, palette=pal)

    cp.set_title(cp.get_xlabel())

    cp.set_xlabel("",visible=False)

    despine()



show()



percent = series.value_counts().apply(lambda x: "{:.2f}%".format(x / len(series) * 100))

percent.name = "Percent"

count = series.value_counts()

count.name = "Count"

output = DataFrame([percent,count])

display(output)
series = data["DayOfWeek"]



set_style("whitegrid")



count = series.value_counts()

count.name = "Count"



pal = {}

for value in data["DayOfWeek"].unique(): 

    if count[value] > 100:

        pal[value] = color_palette("colorblind")[4]

    else:

        pal[value] = color_palette("colorblind")[0]

    

with axes_style({'axes.grid': False}):

    

    cp = countplot(series, order=["Mon","Tues","Wed","Thur","Fri","Sat","Sun"], palette=pal)

    cp.set_title(cp.get_xlabel())

    cp.set_xlabel("",visible=False)

    despine()



show()



percent = series.value_counts().apply(lambda x: "{:.2f}%".format(x / len(series) * 100))

percent.name = "Percent"

output = DataFrame([percent,count])

display(output)
series = data["DayofMonth"]



set_style("whitegrid")



count = series.value_counts()

count.name = "Count"



tier_1 = count[count <= 10].index.tolist()

tier_2 = count[(count > 10 ) & (count < 40)].index.tolist()

tier_3 = count[count >= 40].index.tolist()



pal = {}

for index in tier_1: pal[index] = color_palette("colorblind")[0]

for index in tier_2: pal[index] = color_palette("colorblind")[1]

for index in tier_3: pal[index] = color_palette("colorblind")[4]

    

with axes_style({'axes.grid': False}):

    

    cp = countplot(series, palette=pal)

    cp.set_title(cp.get_xlabel())

    cp.set_xlabel("",visible=False)

    despine()



show()



percent = series.value_counts().apply(lambda x: "{:.2f}%".format(x / len(series) * 100))

percent.name = "Percent"

output = DataFrame([percent,count])

display(output)
def category_x_category_analysis(data, category_a, category_b):

    

    set_style("whitegrid")

    

    with axes_style({'axes.grid': False}):

        cp = countplot(data=data, x=category_a, hue=category_b, palette=color_palette("colorblind"))

        cp.set_title(cp.get_xlabel())

        cp.set_xlabel("",visible=False)

        despine()

    

    show()
category_x_category_analysis(data, "Month", "DayOfWeek")
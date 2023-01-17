from pandas import read_csv

data = read_csv("../input/DelayedFlights.csv")
data.head()
data = data.drop("Unnamed: 0",1)
target = ["Cancelled"]

leaky_features = ["CancellationCode", "Year", "Diverted", "ArrTime", "ActualElapsedTime", "AirTime", "ActualElapsedTime", "AirTime", "ArrDelay", "TaxiIn", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay","LateAircraftDelay"]

features = [x for x in data.columns if (x != target[0]) & (x not in leaky_features) & (len(data[x].unique().tolist()) > 1)]
data = data[data["Month"].isin([10,11,12])]
def get_dtypes(data,features):

    output = {}

    for f in features:

        dtype = str(data[f].dtype)

        if dtype not in output.keys(): output[dtype] = [f]

        else: output[dtype] += [f]

    return output



def show_uniques(data,features):

    for f in features:

        if len(data[f].unique()) < 30:

            print("%s: %s" % (f,data[f].unique()))

        else:

            print("%s: count(%s)" % (f,len(data[f].unique())))



def show_all_uniques(data,features):

    dtypes = get_dtypes(data,features)

    for key in dtypes.keys():

        print("\n" + key + "\n")

        show_uniques(data,dtypes[key])
show_all_uniques(data,features)
dtypes = get_dtypes(data,features)
categories = ["Month", "DayOfWeek", "DayofMonth"]

categories += dtypes["object"]

numerics = [i for i in dtypes["int64"] if i not in categories]

numerics += dtypes["float64"]
data[categories].head()
categories
from itertools import groupby

from numpy import nan



def split_text(text):

    

    sequence = [''.join(unit) for split_point, unit in groupby(text, str.isalpha)]

    

    if len(sequence) < 3:

        

        if sequence[0].isalpha():

            

            sequence += [str(nan)]

            

        else:

            

            sequence = [str(nan)] + sequence

            

        

    return tuple(sequence)



def split_tailnum(data, series):

    

    TailNum_0 = []

    TailNum_1 = []

    TailNum_2 = []

    

    round_by_first_2 = lambda x: round(int(x),2 - len(x))

    

    for value in series: 

        splits = split_text(value)

        TailNum_0 += [splits[0]]

        TailNum_1 += [round_by_first_2(splits[1])]

        TailNum_2 += [splits[2]]

    

    data["TailNum_0"] = TailNum_0

    data["TailNum_1"] = TailNum_1

    data["TailNum_2"] = TailNum_2

    data["TailNum_2_has_AA"] = data["TailNum_2"].apply(lambda x: 1 if "AA" in x else 0)

    

    return data

    

data = split_tailnum(data, data["TailNum"])



categories += ["TailNum_0","TailNum_1","TailNum_2", "TailNum_2_has_AA"]

categories.remove("TailNum")
from seaborn import countplot, set_palette, color_palette, set_style, despine

from matplotlib.pyplot import show, subplot, figure, suptitle
set_style("whitegrid")



for category in categories:

    

    if len(data[category].unique()) < 15:

    

        figure(figsize=(12.5,6))

        

        suptitle(category)

        

        subplot(121)

        ax = countplot(data=data[data[target[0]] == 0], x=category, color="#2c2a2f")

        ax.set_xlabel("", visible=False)

        ax.set_ylabel("", visible=False)

        despine()

        

        subplot(122)

        ax = countplot(data=data[data[target[0]] == 1], x=category, color="#2c2a2f")

        ax.set_xlabel("", visible=False)

        ax.set_ylabel("", visible=False)

        despine()

        

        show()

        

    else:

        

        figure(figsize=(12.5, 0.25 * len(data[category].unique())))

        

        suptitle(category)

        

        order = data[category].value_counts().index.tolist()

        

        subplot(121)

        ax = countplot(data=data[data[target[0]] == 0], y=category, order=order, color="#2c2a2f")

        ax.set_xlabel("", visible=False)

        despine()

        

        subplot(122)

        ax = countplot(data=data[data[target[0]] == 1], y=category, order=order, color="#2c2a2f")

        ax.set_xlabel("",visible=False)

        despine()

        

        show()

        
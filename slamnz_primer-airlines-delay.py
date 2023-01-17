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
# TailNum New Features

"""

from itertools import groupby

from numpy import nan



def split_text(text):

    

    sequence = [''.join(unit) for split_point, unit in groupby(text, str.isalpha)]

    

    if len(sequence) < 3:

        

        if sequence[0].isalpha():

            

            sequence += [nan]

            

        else:

            

            sequence = [nan] + sequence

            

        

    return tuple(sequence)



data["TailNum_0"] = data["TailNum"].apply(lambda x: nan if type(x) == float else split_text(x)[0])

data["TailNum_1"] = data["TailNum"].apply(lambda x: nan if type(x) == float else split_text(x)[1])

data["TailNum_2"] = data["TailNum"].apply(lambda x: nan if type(x) == float else split_text(x)[2])



numerics += ["TailNum_0","TailNum_1","TailNum_2"]

numerics.remove("TailNum")

"""



pass
data[categories].head()
data[numerics].head()
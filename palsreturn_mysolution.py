import pandas as pd
import numpy as np

data = pd.read_csv("../input/uncover/UNCOVER/github/covid-19-uk-historical-data.csv")
attributes = data.columns
countries = np.unique(data[attributes[1]])
areas = data[attributes[2]]
areas[areas.isnull()] = "Nan"
areas = np.unique(areas)
data[attributes[-1]][data[attributes[-1]].isnull()] = 0
data[attributes[-1]][data[attributes[-1]]=='1 to 4'] = 4

total_cases = np.sum(np.asarray(data[attributes[-1]], dtype = int))

print("Total cases " + str(total_cases))
country_wise_cases = dict()

for c in countries:
    t = np.asarray(data[attributes[-1]][data[attributes[1]]==c], dtype = int)
    country_wise_cases[c] = np.sum(t)
    
print(country_wise_cases)
from os import listdir
from os.path import isfile, join
import pandas as pd

mypath = "../input/uncover/UNCOVER/google_mobility/"
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in files:
    t = pd.read_csv(join(mypath,f))
    c = t.columns
    print(f+"->"+str(c))
import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

mob_dataset = "../input/uncover/google_mobility/"
covid_tracking_proj = "../input/uncover/covid_tracking_project"

coviddata = pd.read_csv(join(covid_tracking_proj, "covid-statistics-by-us-states-daily-updates.csv"))
mobdata = pd.read_csv(join(mob_dataset, "us-mobility.csv"))

statecodes = np.unique(coviddata["state"])
states = np.unique(mobdata["state"])

coviddata["positive"][coviddata["positive"].isnull()] = 0

statewise_total_case = {}
for s in statecodes:
    t = np.asarray(coviddata["positive"][coviddata["state"]==s], dtype = int)
    statewise_total_case[s] = t[::-1]

ts = int(math.sqrt(statecodes.shape[0]))

def stationarize(data, alpha):
    logdata = [math.log(x+alpha) for x in data]
    stdata = []
    for i in range(0, len(logdata)-1):
        stdata.append(logdata[i+1]-logdata[i]) 
    return np.array(stdata)

def destationarize(data, alpha):
    D = [data[0]]
    for i in range(1,data.shape[0]):
        D.append(D[i-1]+data[i])
        
    
    for i in range(0, len(D)):
        D[i] = math.exp(D[i])-alpha
    return np.array(D)

def autocorrelation(x):
    res = np.correlate(x, x,  mode = "full")
    return res[res.shape[0] // 2:]

fig, axs = plt.subplots(ts, ts)

t = 0
for i in range(0,ts):
    for j in range(0,ts):
        axs[i,j].plot(autocorrelation(statewise_total_case[statecodes[t]]))
        t = t+1
        if(t>=statecodes.shape[0]):
            break
    if(t>=statecodes.shape[0]):
        break

plt.savefig("/kaggle/working/autocorrelation_us_data.png")




import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

mob_dataset = "../input/uncover/google_mobility/"
covid_tracking_proj = "../input/uncover/covid_tracking_project"

coviddata = pd.read_csv(join(covid_tracking_proj, "covid-statistics-by-us-states-daily-updates.csv"))
mobdata = pd.read_csv(join(mob_dataset, "us-mobility.csv"))

statecodes = np.unique(coviddata["state"])
states = np.unique(mobdata["state"])

coviddata["positive"][coviddata["positive"].isnull()] = 0

statewise_total_case = {}
for s in statecodes:
    t = np.asarray(coviddata["positive"][coviddata["state"]==s], dtype = int)
    statewise_total_case[s] = t[::-1]

ts = int(math.sqrt(statecodes.shape[0]))

def stationarize(data, alpha):
    logdata = [math.log(x+alpha) for x in data]
    stdata = []
    for i in range(0, len(logdata)-1):
        stdata.append(logdata[i+1]-logdata[i]) 
    return np.array(stdata)

def destationarize(data, alpha):
    D = [data[0]]
    for i in range(1,data.shape[0]):
        D.append(D[i-1]+data[i])
        
    
    for i in range(0, len(D)):
        D[i] = math.exp(D[i])-alpha
    return np.array(D)

def autocorrelation(x):
    res = np.correlate(x, x,  mode = "full")
    return res[res.shape[0] // 2:]

stdata = stationarize(statewise_total_case[statecodes[0]],0.001)
model = ARIMA(stdata, order = (1,0,1))
results = model.fit(disp=-1, solver = 'nm')
print(str(stdata[0])+" "+str(results.predict(start = 0)[0]))

plt.plot(stdata, color = "green")
plt.plot(results.fittedvalues, color = "blue")
#plt.plot(statewise_total_case[statecodes[0]], color = "red")
plt.show()

import pandas as pd
import numpy as np
from os.path import join
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mob_dataset = "../input/uncover/google_mobility/"
covid_tracking_proj = "../input/uncover/covid_tracking_project"

coviddata = pd.read_csv(join(covid_tracking_proj, "covid-statistics-by-us-states-daily-updates.csv"))
mobdata = pd.read_csv(join(mob_dataset, "us-mobility.csv"))

statecodes = np.unique(coviddata["state"])
states = np.unique(mobdata["state"])

coviddata["positive"][coviddata["positive"].isnull()] = 0

statewise_total_case = {}
for s in statecodes:
    t = np.asarray(coviddata["positive"][coviddata["state"]==s], dtype = int)
    statewise_total_case[s] = t[::-1]

ts = int(math.sqrt(statecodes.shape[0]))

def stationarize(data, alpha):
    logdata = [math.log(x+alpha) for x in data]
    stdata = []
    for i in range(0, len(logdata)-1):
        stdata.append(logdata[i+1]-logdata[i]) 
    return np.array(stdata)

def destationarize(data, alpha):
    D = [data[0]]
    for i in range(1,data.shape[0]):
        D.append(D[i-1]+data[i])
        
    
    for i in range(0, len(D)):
        D[i] = math.exp(D[i])-alpha
    return np.array(D)

def createdataset(data, m):
    X = []
    Y = []
    for i in range(0, len(data)):
        x = np.zeros(m)
        k = 0
        for j in range(i-1,i-m-1,-1):
            if(j>=0 and k<m):
                x[k] = data[j]
                k = k+1
        X.append(x)
        Y.append(data[i])
    return np.array(X), np.array(Y)

Dt = stationarize(statewise_total_case[statecodes[1]], 0.0001)
X, Y = createdataset(Dt, 3)
Xtr, Xtest, Ytr, Ytest = train_test_split(X,Y, test_size=0.3)
#model = LinearRegression()
model = MLPRegressor(hidden_layer_sizes=(100,10,10,5))
#cv_score = cross_val_score(model, X, Y, cv = 10)
#print(np.mean(cv_score),np.std(cv_score))
model.fit(Xtr,Ytr)
py = model.predict(Xtest)

py = destationarize(py, 0.0001)
Ytest = destationarize(Ytest, 0.0001)
print(mean_squared_error(py, Ytest))
plt.plot(Ytest, "b")
plt.plot(py, "g")

import pandas as pd
import numpy as np
from os.path import join
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mob_dataset = "../input/uncover/google_mobility/"
covid_tracking_proj = "../input/uncover/covid_tracking_project"

coviddata = pd.read_csv(join(covid_tracking_proj, "covid-statistics-by-us-states-daily-updates.csv"))
mobdata = pd.read_csv(join(mob_dataset, "us-mobility.csv"))

def removenan(df):
    for k in df.keys():
        df[k].fillna(value = 0)
    return df

mobdata = removenan(mobdata)

statecodes = np.unique(coviddata["state"])
states = np.unique(mobdata["state"])
code_state = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", 
                  "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL" : "Florida", "GA": "Georgia", 
                  "HI" : "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
                  "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", 
                  "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", 
                  "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
                  "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", 
                  "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", 
                  "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",}

coviddata["positive"][coviddata["positive"].isnull()] = 0

statewise_total_case = {}
for s in statecodes:
    t = np.asarray(coviddata["positive"][coviddata["state"]==s], dtype = int)
    statewise_total_case[s] = t[::-1]

ts = int(math.sqrt(statecodes.shape[0]))

def createdataset(stdata, mobdata, m):
    X = []
    Y = []
    
    for key in stdata.keys():
        data = stdata[key]
        if key in code_state.keys():
            wp = np.array(mobdata[mobdata['state']==code_state[key]]['workplaces'])
        else:
            wp = np.array([])
        for i in range(0, len(data)):
            x = np.zeros(m)
            k = 0
            for j in range(i-1,i-m-1,-1):
                if(j>=0 and k<m):
                    x[k] = data[j]
                    k = k+1            
            if i<wp.shape[0]:
                xt = np.append(x,wp[i])
            else:
                xt = np.append(x, 0)
            X.append(xt)
            Y.append(data[i])
    return np.array(X), np.array(Y)

X, Y = createdataset(statewise_total_case, mobdata, 4)


Xtr, Xtest, Ytr, Ytest = train_test_split(X,Y, test_size=0.3)
model = LinearRegression()
#cv_score = cross_val_score(model, X, Y, cv = 10)
#print(np.mean(cv_score),np.std(cv_score))
model.fit(Xtr,Ytr)
py = model.predict(Xtest)

print(mean_squared_error(py, Ytest))
plt.plot(Ytest, "b")
plt.plot(py, "g")

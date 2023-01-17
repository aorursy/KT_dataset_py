import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
def date_to_jd(date):

#transform date (either julian or gregorian) into a julian day number

    y = date[0]

    mo = date[1]

    d = date[2]

    h = date[3]

    mn = date[4]

    s = date[5]



    if date[:3] >= [1582,10,15]:

        #gregorian date

        return 367*y - (7*(y+int((mo+9)/12)))//4 - (3*(int((y+(mo-9)/7)/100)+1))//4+(275*mo)//9+d+1721028.5+h/24+mn/(24*60)+s/86400

    elif date[:3] <= [1582,10,4]:

        #julian date

        return 367*y - (7*(y+5001+int((mo-9)/7)))//4+(275*mo)//9+d+1729776.5+h/24+mn/(24*60)+s/86400



def jd_to_date(jd):

    Z = int(jd+0.5)

    F = (jd+0.5)%1

    if Z < 2299161:

        A = Z

    else:

        g = int((Z - 1867216.25) / 36524.25)

        A = Z + 1 + g - g//4 



    B = A + 1524

    C = int((B-122.1) / 365.25)

    D = int(365.25 * C)

    E = int((B-D) / 30.6001)

 

    d = B - D - int(30.6001*E) + F

    if E<14:

        mo = E-1

    else:

        mo = E-13    



    if mo >2:

        y = C- 4716

    else:

        y = C - 4715

    

    return str(y)+'-'+mak_2_dig(mo)+'-'+mak_2_dig(int(d))
def mak_2_dig(x):

#transforms all integers bewteen 0 and 99 into a 2-digit string

    if x<10:

        return '0'+str(x)

    else:

        return str(x)





def transf_date(s):

    s = s.split()

    s[1] = str(dic_months[s[1]])

    return s[0]+':'+s[1]+':'+s[2]



dic_months = {'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June': 6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
df = pd.read_csv("../input/solar.csv")

df = df.loc[:, ['Calendar Date', 'Eclipse Time', 'Eclipse Type']]

#combine Date and Time to Time JD

df['Calendar Date'] = df['Calendar Date'].apply(lambda x:transf_date(x))

df['Time'] = df.loc[:,['Calendar Date', 'Eclipse Time']].apply(lambda x: x[0]+':'+x[1], axis = 1) 

df = df.drop(['Calendar Date', 'Eclipse Time'], axis=1)

df['Time JD']=df['Time'].apply(lambda x : date_to_jd([int(j) for j in x.split(':')]))

del df['Time']
df = df[df['Eclipse Type'].str[0] != 'P']     #exclude partial eclipses

##possibly exclude data prior to date

#date_start = date_to_jd([1400, 1, 1, 0, 0, 0])   #only use data after this date

#df = df[df['Time JD'] > date_start]

t_between = df['Time JD'].diff().tolist()   #count days between consecutive eclipses

t_between = t_between[1:-1]                 #drop first and last NaN
plt.hist(t_between,int(max(t_between))+1)

plt.xlabel('Time (days)')

plt.show()
set([int(d) for d in t_between])
today = date_to_jd([2017, 3, 1, 0, 0,0])



df_before = df[df['Time JD'] <= today]

df_after = df[df['Time JD'] > today]

dates_before = df_before['Time JD'].tolist()

dates_after = df_after['Time JD'].tolist()

diff_before = [int(j) for j in t_between[:len(dates_before)-1]]  #recorded differences between past ecl.

diff_after =  [int(j) for j in t_between[len(dates_before)-1:]]   #differences betw. future ecl. to be predicted
L = len(diff_before)//82
X = []

y = []

for j in range(len(diff_before)-L):

    X.append(diff_before[j:j+L])

    y.append(diff_before[j+L]) 

 

X = np.array(X)

y = np.array(y)
p = 0.15 #fraction used for validation

X_val = X[int((1-p)*len(X)):]

y_val = y[int((1-p)*len(y)):]



X_train = X[:int((1-p)*len(X))]

y_train = y[:int((1-p)*len(y))]
0#lrn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,100), learning_rate = 'adaptive')

#lrn = SVC(kernel = 'linear', C=1000)     #just takes too long

lrn = RandomForestClassifier(1000)

lrn.fit(X_train, y_train)
c = 0

s = lrn.predict(X_val)

for i in range(len(y_val)):

    if np.absolute(y_val[i] - s[i]) > 1:

        c += 1       #count wrong predictions

print("Predictions more than 1 day off: {}%".format(((c*1000)//len(y_val))/10))

print("Tested on {} samples".format(len(y_val)))
lrn = xgb.XGBClassifier(learning_rate = 0.05, max_depth = 10, objective = "reg:linear")

lrn.fit(X_train, y_train)



c = 0

s = lrn.predict(X_val)

for i in range(len(y_val)):

    if np.absolute(y_val[i] - s[i]) > 1:

        c += 1       #count wrong predictions

print("Predictions more than 1 day off: {}%".format(((c*1000)//len(y_val))/10))

print("Tested on {} samples".format(len(y_val)))
N_future = 10   #number of predictions into the future

lrn.fit(X, y)   # optionally train the model again on the 'entire' past 

#prediction part: append every further prediction to the feature set



xx = np.array(X[-1])

xx = np.roll(xx,-1)

xx[-1] = y[-1]

y_pred = []

d_last = dates_before[-1]



for i in range(N_future):

    yy = lrn.predict([xx])

    y_pred.append(yy[0])

    xx = np.roll(xx,-1)

    xx[-1] = yy





print("Prediction:      ", y_pred)

print("Calculated/true: ", diff_after[:N_future])
days_pred = [d_last + i for i in np.cumsum(y_pred)]

print("Date (predicted)   Date (calculated/true)")

for i in range(len(days_pred)):

    print("  "+jd_to_date(days_pred[i])+"          "+jd_to_date(dates_after[i]))
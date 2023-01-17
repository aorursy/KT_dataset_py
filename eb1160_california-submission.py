import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

from datetime import timedelta

import scipy

import sklearn



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

df.head()
df['datetime'] = pd.to_datetime(df.Date)

df.index = df.datetime
## According to the US Census Bureau

## https://www.census.gov/quickfacts/CA

ca_population = 39512223 # As of July 2019



## Coronavirus contact rate

## According to this article from Reuters

## https://www.reuters.com/article/us-china-health-transmission/coronavirus-contagion-rate-makes-it-hard-to-control-studies-idUSKBN1ZO0QW

beta = 1/2.5 ## Based on a study from Britain's Lancaster University



## Coronavirus recovery rate from California

## Calculated based on demographic factors from South Korea data

## gamma = 





## Most frequent Symptom Lag

## According to BBC

## https://www.bbc.com/news/health-51800707

t = 5



## Ratio of tests per member of the population in South Korea to the same figure in the US

## https://www.npr.org/sections/coronavirus-live-updates/2020/03/24/820981710/fact-check-u-s-testing-still-isnt-close-to-what-south-korea-has-done

testing_ratio = 1090/170
df['ActualCases'] = df.ConfirmedCases.apply(lambda x: x* testing_ratio)
df
import matplotlib.pyplot as plt

df['datetime'] = pd.to_datetime(df.datetime)

firstcase = df.loc[df.ConfirmedCases>=0].datetime.min()

df1 = df.loc[df.datetime>firstcase ]





plt.rcParams["figure.figsize"] = [6.4*3, 4.8*3]

plt.plot(df1.ConfirmedCases, color = 'grey', label = 'Confirmed Cases')

plt.plot(df1.Fatalities, color = 'red', label = 'Fatalities')

plt.plot(df1.ActualCases, color = 'orange', linestyle = '--', label = 'Potential Actual Cases')

plt.legend()

# plt.yscale('log')

plt.show()



first_confirmed_case = df.loc[df.ConfirmedCases>0].index.min()

df1 = df.loc[df.index>=first_confirmed_case]

print(first_confirmed_case)
## Mortality rate of those tested

def newdiv(n1, n2):

    if n2!=0:

        return n1/n2

    else:

        return 0

    

mort = df1.apply(lambda x: newdiv(x.Fatalities, x.ConfirmedCases), axis = 1)

m,b = np.polyfit([i for i in range(len(mort))], mort, 1)

print(m)

print(b)

plt.scatter(x = df1.index, y = mort)

best_fit = pd.Series([i*m+b for i in range(len(df1.index))])

best_fit.index = df1.index

plt.title('Changes in COVID-19 Mortality Rate Since First Confirmed Cases')

plt.xlabel('date')

plt.ylabel('mortality rate')

plt.plot(best_fit, color = 'red', linestyle = '--')



plt.show()
import os

for dirname, _, filenames in os.walk('/kaggle/input/coronavirusdataset'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
routes = pd.read_csv('/kaggle/input/coronavirusdataset/PatientRoute.csv')

province = pd.read_csv('/kaggle/input/coronavirusdataset/TimeProvince.csv')

regions = pd.read_csv('/kaggle/input/coronavirusdataset/Region.csv')

age = pd.read_csv('/kaggle/input/coronavirusdataset/TimeAge.csv')

patient = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')

gender = pd.read_csv('/kaggle/input/coronavirusdataset/TimeGender.csv')

time = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')
time.head()

time.index = pd.to_datetime(time.date)

plt.plot(time.test, color = 'grey',label = 'tested')

plt.plot(time.confirmed, color = 'teal', label = 'confirmed')

plt.plot(time.released, color = 'purple', label = 'released')

plt.plot(time.deceased, color = 'yellow', label = 'deceased')

plt.legend()

plt.title('Spread of COVID-19 in South Korea')

plt.xlabel('date')

plt.ylabel('count')

plt.yscale('log')

plt.show()
patient.head()

patient['symptom_onset_date'] = pd.to_datetime(patient['symptom_onset_date'])

patient['confirmed_date'] = pd.to_datetime(patient['confirmed_date'])

patient['released_date'] = pd.to_datetime(patient['released_date'])

patient['deceased_date'] = pd.to_datetime(patient['deceased_date'])

patient['onset_to_confirmed']  = patient.apply(lambda x: x.confirmed_date - x.symptom_onset_date, axis = 1)

patient['onset_to_mortality'] = patient.apply(lambda x: x.confirmed_date - x.symptom_onset_date, axis = 1)

patient['confirmed_to_mortality'] = patient.apply(lambda x: x.deceased_date - x.confirmed_date, axis = 1)

patient['onset_to_released'] = patient.apply(lambda x: x.released_date - x.symptom_onset_date, axis = 1)

patient['confirmed_to_released'] = patient.apply(lambda x: x.released_date - x.confirmed_date, axis = 1)
ovs = patient.loc[patient.infection_case == 'overseas inflow']

dates_of_overseas_confirmed = list(set(ovs.confirmed_date.dt.date.tolist()))

dates_of_overseas_confirmed.sort()
pc_dict = province.pivot_table(index = ['date'], columns = 'province', values = 'confirmed', aggfunc = np.sum)

pr_dict = province.pivot_table(index = ['date'], columns = 'province', values = 'released', aggfunc = np.sum)

pd_dict = province.pivot_table(index = ['date'], columns = 'province', values = 'deceased', aggfunc = np.sum)
province_piv = province.pivot_table(index = ['date'], columns = 'province', values = 'confirmed', aggfunc = np.sum)

cases_by_province = {}

for c in province_piv.columns:

    cases_by_province[c] = province_piv[c].to_dict()

    plt.plot(province_piv[c], label = c)



plt.title('Confirmed Cases of COVID-19 in South Korea by Province')

plt.xlabel('date')

plt.ylabel('Confirmed Cases by Province')

plt.yscale('log')

plt.legend()

plt.xticks(rotation = 45)

plt.show()


def cases_map(pdict, pr, date):

    try:

        return pdict[pr][str(date.date())]

    except:

        return np.nan



patient['cases_in_province'] = patient.apply(lambda x: cases_map(pc_dict,x.province, x.confirmed_date), axis = 1)

patient['age_years'] = (datetime.datetime.now().year-patient.birth_year)
hlth.iloc[0].values
r1 = regions.pivot_table(index = 'province',values = ['university_count'], aggfunc = np.sum).to_dict()

r2 = regions.pivot_table(index = 'province',values = ['elderly_population_ratio'], aggfunc = np.mean).to_dict()

# if 'date' in province.columns:

#     province.drop('date', inplace = True, axis = 1)

if 'time' in province.columns:

    province.drop('time', inplace = True, axis = 1)

province['average_elderly_ratio'] = province.province.map(r2['elderly_population_ratio'])

province['university_count'] = province.province.map(r1['university_count'])
hlth = pd.read_csv('/kaggle/input/south-korea-number-of-health-centers/Number_of_Health_Center__Health_Center_Branch__Health_Care_Center__1997____20200326033038.csv')

hlth[['By Si-Do(1)','2018',

       '2018.1', '2018.2', '2018.3', '2018.4']]

hlth.columns = hlth.iloc[0]

hlth.index = hlth['By Si-Do(1)']

hlth.drop('By Si-Do(1)', axis = 0, inplace = True)

hlth.drop('By Si-Do(1)', axis = 1, inplace = True)



hc_dict = hlth['Health Care Center'].to_dict()['Health Care Center']



province['province'] = province.province.apply(lambda x: x.replace('-do','').replace('sang','').replace('cheong','').replace('lla','').replace('Jeobuk','Jeonbuk').replace('Jeonam','Jeonnam'))

set(province.province.unique())-set(hc_dict.keys())

province['health_care_centers'] = province.province.map(hc_dict)
province
patient['cases_in_province_t5'] = patient.cases_in_province.shift(5)

province['cases_in_province_t5'] = province.confirmed.shift(5)

patient.cases_in_province_t5.fillna(0, inplace = True)

province.cases_in_province_t5.fillna(0, inplace = True)



patient.head()

pdfr = patient.loc[patient.released_date.notna()][['sex','age_years','onset_to_released','cases_in_province','cases_in_province_t5']]

pdfm = patient.loc[patient.deceased_date.notna()][['sex','age_years','confirmed_to_mortality','cases_in_province','cases_in_province_t5']]
plt.title("South Korea Onset to Mortality by Gender and Age")

plt.scatter(x = pdfm.loc[pdfm.sex == 'male'].age_years,y = pdfm.loc[pdfm.sex == 'male'].confirmed_to_mortality.dt.days, color = 'blue')

plt.scatter(x = pdfm.loc[pdfm.sex == 'female'].age_years,y = pdfm.loc[pdfm.sex == 'female'].confirmed_to_mortality.dt.days, color = 'pink')
province.index = pd.to_datetime(province.date)



for p in province.province.unique():

    pr = province.loc[province.province == p]

    plt.plot(pr.confirmed, color = 'teal')

    plt.plot(pr.released, color = 'yellow')

    plt.plot(pr.deceased, color = 'grey')

    

plt.show()
province['province'] = province.province.apply(lambda x: x.replace('-do','').replace('Jeobuk','Jeonbuk').replace('Jeonam','Jeonnam').replace('sang','').replace('cheong','').replace('lla',''))

set(province.province.unique())-set(hc_dict.keys())

province['health_care_centers'] = province.province.map(hc_dict)

province
## Calculate Days Since Initial Outbreak



first_case = {}

for p in province.province.unique():

    pr = province.loc[(province.province == p)&(province.confirmed>0)]

    fd = pr.index.min()

    first_case[p] = fd
province
province
province['first_case'] = province.province.map(first_case)

province.first_case

province['time_since_first_case'] = (pd.to_datetime(province.date)-pd.to_datetime(province.first_case)).dt.days

province.drop('first_case', inplace = True, axis = 1)
province.columns

province_data = {}



'''

The goal is to end up with the coefficients of the infection, mortality and recovery curves, and then predict for the entire state of California

'''

pr_columns_all = province.columns

pr_columns = [

       'average_elderly_ratio', 'university_count', 'health_care_centers',

       'cases_in_province_t5', 'time_since_first_case']

pr_indexes = {}

pr_confirmed = {}

pr_released = {}

pr_deceased = {}

province_data_truncated = {}





for p in province.province.unique():

    pr = province.loc[province.province == p][['confirmed','released','deceased',

       'average_elderly_ratio', 'university_count', 'health_care_centers',

       'cases_in_province_t5', 'time_since_first_case']]

    pr2 = pr.loc[pr.time_since_first_case>=0]

    pr_indexes[p] = pr2.index

    pr_confirmed[p] = pr2.confirmed.values

    pr_released[p] = pr2.released.values

    pr_deceased[p] = pr2.deceased.values 

    province_data[p] = pr[['average_elderly_ratio', 'university_count', 'health_care_centers',

       'cases_in_province_t5', 'time_since_first_case']].values

    province_data_truncated[p] = pr2[['average_elderly_ratio', 'university_count', 'health_care_centers',

       'cases_in_province_t5', 'time_since_first_case']].values
c = model.coef_[0]

b = model.intercept_





vs = []

for u in x:

    v = c[0]*u**4 + c[1]*u**3 + c[2]*u**2 + c[3]*u**1+b

    vs.append(v)

    

plt.scatter(x=x,y = vs)
# from sklearn.utils.extmath import safe_sparse_dot

# cmatrix = np.dot(np.array(x), model.coef_.T.reshape(1,-1))+ model.intercept_

# model.densify()
# model.coef_.T


import operator



import numpy as np

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures

'''

Code used originally from Animesh Agarwal, via

https://towardsdatascience.com/polynomial-regression-bbe8b9d97491



'''

r2_values = []

coeffs = {}



for p in pr_confirmed.keys():

    print(p)

    y = pr_confirmed[p]

    x = province_data_truncated[p][:,4]

    # transforming the data to include another axis

    x = x[:, np.newaxis]

    y = y[:, np.newaxis]



    polynomial_features= PolynomialFeatures(degree=3)

    x_poly = polynomial_features.fit_transform(x)

    



    model = LinearRegression()

    model.fit(x_poly, y)

    y_poly_pred = model.predict(x_poly)

    

#     data = pd.DataFrame.from_dict({

#     'x': x,

#     'y': y

#         })



#     p = PolynomialFeatures(degree=3).fit(data)

#     features = DataFrame(p.transform(data), columns=p.get_feature_names(data.columns))





    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))

    r2 = r2_score(y,y_poly_pred)

    r2_values.append(r2)

    print(rmse)

    print(r2)



    plt.scatter(x, y, s=10)

    # sort the values of x before line plot

    sort_axis = operator.itemgetter(0)

    sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)

    x, y_poly_pred = zip(*sorted_zip)

    

    c = model.coef_[0]

    b = model.intercept_



    vs = []

    for u in x:

        v = c[0]*u**0 + c[1]*u**1 + c[2]*u**2 + c[3]*u**3+b[0]

        vs.append(v)



    coeffs[p] = [b[0], c[0],c[1],c[2],c[3]]

    plt.plot(x, y_poly_pred, color='m')

    plt.plot(vs, color = 'yellow', linestyle = '--')

    plt.show()
province['b']=province.province.map(coeffs).apply(lambda x: x[0])

province['c0']=province.province.map(coeffs).apply(lambda x: x[1])

province['c1']=province.province.map(coeffs).apply(lambda x: x[2])

province['c2']=province.province.map(coeffs).apply(lambda x: x[3])

province['c3'] = province.province.map(coeffs).apply(lambda x: x[4])
f = province[[

       'average_elderly_ratio', 'university_count', 'health_care_centers',

       'cases_in_province_t5']].values



t1 = province.b.values

t2 = province.c1.values

t3 = province.c2.values

t4 = province.c3.values


count = 1

for i in range(f.shape[1]):

#     print('\n')

#     print(i)

    f_ = f[:,i]

    for t in [t1,t2,t3,t4]:

#         print(count)

        model = LinearRegression()

        model.fit(f_.reshape(-1,1), t)

        y_poly_pred = model.predict(f_.reshape(-1,1))

        count +=1

        rmse = np.sqrt(mean_squared_error(t,y_poly_pred))

        r2 = r2_score(t,y_poly_pred)

#         print(r2)



# for t in [t1,t2,t3,t4]:

#     print(count)

#     model = LinearRegression()

#     model.fit(f, t)

#     y_poly_pred = model.predict(f)

#     count +=1

#     rmse = np.sqrt(mean_squared_error(t,y_poly_pred))

#     r2 = r2_score(t,y_poly_pred)

#     print(r2)
regions.university_count.sum()
province.columns

provinces_labels = province.province

provinces['health_care_centers'] = province.health_care_centers(lambda x: float(x))

korea_vectors = [province[['health_care_centers','university_count', 'average_elderly_ratio']].values[i] for i in range(len(province))]
hospitals = 81729/58

unis = 138/58

elderly_ratio  = 14.3



ca_vector = [hospitals, unis, elderly_ratio]

distances = {}

count = 0



for v in korea_vectors:

    v = [float(i) for i in v]

    dist = scipy.spatial.distance.cosine(v, ca_vector)

    distances[provinces_labels[count]] = dist

    count +=1

sim = pd.Series(distances)

sim.sort_values(ascending = False, inplace = True)

sim
df.columns
coeffs = province.loc[province.province == 'Seoul'][['b','c0','c1','c2','c3']].drop_duplicates()

b = coeffs['b']

c1 = coeffs['c1']

c2 = coeffs['c2']

c3 = coeffs['c3']
test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

train_and_test = pd.concat([df[['Date','ConfirmedCases','Fatalities','ActualCases']], test[['ForecastId','Date']]])

test.head()
train_and_test['day_number'] = np.array(range(len(train_and_test)))

train_and_test['calculated_confirmed'] = train_and_test.day_number.apply(lambda x: (b+c1*x**1+c2*x**2+c3*x**3) )

train_and_test['calculated_actual'] = train_and_test.calculated_confirmed.apply(lambda x: x*testing_ratio)

# train_and_test['calculated_fatalities'] = train_and_test.day_number.apply(lambda x: )
m,b = np.polyfit([i for i in range(len(mort))], mort, 1)

print(m,b)

train_and_test['mortality_rate'] = [i*m+b for i in range(len(train_and_test))]

train_and_test['mortalities'] = train_and_test.apply(lambda x: float(x.mortality_rate)*float(x.calculated_actual), axis = 1)
train_and_test.mortality_rate
train_and_test[['mortality_rate', 'calculated_actual']]
plt.title('Seoul vs. California Predicted Cases')

plt.plot(train_and_test.calculated_confirmed.values, color = 'blue', linestyle = '--')

plt.plot(train_and_test.calculated_actual.values, color = 'orange', linestyle = '--',label = 'calculated_actual''')

plt.plot(train_and_test.mortalities.values, color = 'red', linestyle = '--', label = 'mortalities')

plt.plot(train_and_test.ConfirmedCases.values, color = 'teal', label = 'Confirmed Cases')

plt.legend

plt.show()

train_and_test.columns
results = train_and_test[['ForecastId','calculated_confirmed','mortalities']]

results['ConfirmedCases'] = results['calculated_confirmed']

results['Mortalities'] = results['mortalities']
results.to_csv('submission.csv')
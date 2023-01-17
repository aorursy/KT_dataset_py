import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport





import os 

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv(r'/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')
train.head()
test =  pd.read_csv(r'/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')
test.head()
submit = pd.read_csv(r'/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')
submit.head()
covid_19_data = pd.read_csv(r'/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid_19_data.head()
COVID19_line_list_data = pd.read_csv(r'/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
COVID19_line_list_data.head()
COVID19_open_line_list = pd.read_csv(r'/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')
COVID19_open_line_list.head()
time_series_covid19_confirmed = pd.read_csv(r'/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
time_series_covid19_confirmed.head()
time_series_covid19_deaths = pd.read_csv(r'/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
time_series_covid19_deaths.head()
time_series_covid19_recovered = pd.read_csv(r'/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
time_series_covid19_recovered.head()
covid19_italy_province = pd.read_csv(r'/kaggle/input/covid19-in-italy/covid19_italy_province.csv')
covid19_italy_province.head()
covid19_italy_region = pd.read_csv(r'/kaggle/input/covid19-in-italy/covid19_italy_region.csv')
covid19_italy_region.head()
CORD19_research_challenge=pd.read_csv(r'/kaggle/input/CORD-19-research-challenge/metadata.csv')
CORD19_research_challenge.head()
case=pd.read_csv(r'/kaggle/input/coronavirusdataset/Case.csv')
case.head()
patientinfo=pd.read_csv(r'/kaggle/input/coronavirusdataset/PatientInfo.csv')
patientinfo.head()
patientroute=pd.read_csv(r'/kaggle/input/coronavirusdataset/PatientRoute.csv')
patientroute.head()
region=pd.read_csv(r'/kaggle/input/coronavirusdataset/Region.csv')
region.head()
searchtrend=pd.read_csv(r'/kaggle/input/coronavirusdataset/SearchTrend.csv')
searchtrend.head()
seoulfloating=pd.read_csv(r'/kaggle/input/coronavirusdataset/SeoulFloating.csv')
seoulfloating.head()
time=pd.read_csv(r'/kaggle/input/coronavirusdataset/Time.csv')
time.head()
timeage=pd.read_csv(r'/kaggle/input/coronavirusdataset/TimeAge.csv')
timeage.head()
timegender=pd.read_csv(r'/kaggle/input/coronavirusdataset/TimeGender.csv')
timegender.head()
timeprovince=pd.read_csv(r'/kaggle/input/coronavirusdataset/TimeProvince.csv')
timeprovince.head()
weather=pd.read_csv(r'/kaggle/input/coronavirusdataset/Weather.csv')
weather.head()
profile0 = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile0
profile1 = ProfileReport(test, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile1
profile2 = ProfileReport(submit, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile2
profile3 = ProfileReport(covid_19_data, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile3
profile4 = ProfileReport(COVID19_line_list_data, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile4
profile5 = ProfileReport(COVID19_open_line_list, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile5
profile9 = ProfileReport(covid19_italy_province, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile9
profile10 = ProfileReport(covid19_italy_region, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile10
profile11 = ProfileReport(CORD19_research_challenge, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile11
profile12 = ProfileReport(case, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile12
profile13 = ProfileReport(patientinfo, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile13
profile14 = ProfileReport(patientroute, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile14
profile15 = ProfileReport(region, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile15
profile16 = ProfileReport(searchtrend, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile16
profile17 = ProfileReport(seoulfloating, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile17
profile18 = ProfileReport(time, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile18
profile19 = ProfileReport(timeage, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile19
profile20 = ProfileReport(timegender, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile20
profile21 = ProfileReport(timeprovince, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile21
profile22 = ProfileReport(weather, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile22
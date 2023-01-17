import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pycountry
actual_confirmed = np.array([2, 3, 11, 1, 8, 6, 5])
y = np.array([5.110081, 4.744053, 4.776231, 5.211867, 6.185602, 6.603675, 5.165030])
1 - (np.sum(np.square(actual_confirmed - y)) / np.sum(np.square(actual_confirmed)))
# 질병관리본부 해외유입 확진자 데이터(수기 입력)
# https://www.cdc.go.kr/board/board.es?mid=a20501000000&bid=0015
PATH_CONFIRMED = '../input/postcorona/covid_confirmed.csv'

# 로밍 데이터(대회 제공)
PATH_ROAMING1 = '../input/postcorona/2. Roaming_data_1.csv'
PATH_ROAMING2 = '../input/postcorona/2. Roaming_data_2.csv'
PATH_ROAMING3 = '../input/postcorona/2. Roaming_data_3.csv'

# 뉴스 데이터(대회 제공)
PATH_NEWS1 = '../input/postcorona/3-1. NewsList_1.xls'
PATH_NEWS2 = '../input/postcorona/3-1. NewsList_2.xls'
PATH_NEWS3 = '../input/postcorona/3-1. NewsList_3.xls'

# ISO국가코드 -> 대륙 코드 변환 테이블
# https://datahub.io/JohnSnowLabs/country-and-continent-codes-list
PATH_ISO = '../input/postcorona/iso_to_continent.csv'
PATH_CONTINENT = '../input/postcorona/country-and-continent-codes-list-csv_csv.csv'

# 입국 국제항공편 데이터
# http://www.airportal.co.kr/life/airinfo/RbHanFrmMain.jsp
PATH_AIRPORT_SCHEDULE = '../input/postcorona/air_schedule.csv'

# IATA 코드 -> 국가 코드 변환 테이블
# https://datahub.io/core/airport-codes
PATH_IATA = '../input/postcorona/airport-codes_csv.csv'

# 국가별 확진자/사망자/검사 수 데이터
# https://ourworldindata.org/coronavirus
PATH_WORLD_CONFIRMED = '../input/postcorona/daily-cases-covid-19.csv'
PATH_WORLD_DEATHS = '../input/postcorona/daily-deaths-covid-19.csv'
PATH_WORLD_TESTS = '../input/postcorona/full-list-covid-19-tests-per-day.csv'
# 로밍데이터 처리
data = pd.concat([pd.read_csv(PATH_ROAMING1), pd.read_csv(PATH_ROAMING2), pd.read_csv(PATH_ROAMING3)], axis=0)
data.dropna(inplace=True)
data.rename(columns={'count':'n_roamer'}, inplace=True)
data['return'] = pd.to_datetime(data['return'], format='%Y%m%d')
data['arrival'] = pd.to_datetime(data['arrival'], format='%Y%m%d')
data['departure'] = pd.to_datetime(data['departure'], format='%Y%m%d')
data.rename(columns={'return':'date'}, inplace=True)
display(data)

# arrival ~ departure -> 체류기간 정보로 변환
data['d_stay'] = data['departure'] - data['arrival']
data['d_stay'] = data['d_stay'].apply(lambda x: x.days+1) * data['n_roamer']
data.drop(columns=['arrival', 'departure'], inplace=True)


# 로밍국가 단위와 질병관리본부 정례브리핑 국가의 단위를 통합
# iso코드 -> 대륙코드 변환
iso_to_cont = pd.read_csv(PATH_ISO, index_col=0).set_index('iso')
iso_to_cont.loc['cn', 'continent'] = 'China'
iso_to_cont['continent'].replace('SA', 'AM', inplace=True)
iso_to_cont['continent'].replace('NA', 'AM', inplace=True)
data['continent'] = data['iso'].apply(lambda x: iso_to_cont.loc[x, 'continent'])

# 일자별, 대륙별 로밍정보 통합
data = data.groupby(['date', 'continent']).sum()
data
# 확진자/사망자/검사 수 데이터 처리
# 3글자 나라코드를 대륙 코드로 변환
code_to_cont = pd.read_csv(PATH_CONTINENT).set_index('Three_Letter_Country_Code')
code_to_cont = code_to_cont.loc[~code_to_cont.index.duplicated(keep='first')]
code_to_cont.loc['CHN', 'Continent_Code'] = 'China'
code_to_cont['Continent_Code'].fillna(value='AM', inplace=True)
code_to_cont['Continent_Code'].replace('SA', 'AM', inplace=True)
display(code_to_cont.head(5))

daily_fname = {'confirmed':PATH_WORLD_CONFIRMED,
              'deaths':PATH_WORLD_DEATHS,
              'tests':PATH_WORLD_TESTS}
def get_daily_info(info, fname):
    daily_covid = pd.read_csv(fname).dropna()
    daily_covid = daily_covid[daily_covid['Code'].apply(lambda x: len(x)<4)]
    
    # 대륙코드 컬럼 생성
    daily_covid.columns = ['entity', 'code', 'date', info]
    daily_covid['continent'] = daily_covid['code'].apply(lambda x: code_to_cont.loc[x, 'Continent_Code'])
    
    # 날짜 타입 변환
    daily_covid['date'] = pd.to_datetime(daily_covid['date'], format='%b %d, %Y')
    
    # 날짜별, 대륙별 정보 수 합산
    daily_covid = daily_covid[['date', 'continent', info]]
    daily_covid = daily_covid.groupby(['date', 'continent']).sum().reset_index()
    print(f'↓↓↓↓{info}↓↓↓↓')
    display(daily_covid.head())
    
    return daily_covid

for info, fname in daily_fname.items():
    data = data.merge(get_daily_info(info, fname), on=['date', 'continent'], how='left')
data
# 뉴스 데이터
news = pd.concat([pd.read_excel(PATH_NEWS1),
                  pd.read_excel(PATH_NEWS2),
                  pd.read_excel(PATH_NEWS3)], axis=0)
news.columns=['file_name', 'title', 'URL', 'gather_day', 'date', 'channel', 'disease', 'group']
news['date'] = pd.to_datetime(news['date'], format='%Y-%m-%d')
news['date'] = news['date'].apply(lambda x:x.date()).astype('datetime64')

# COVID-19 기사 합계
n_news1 = news[news['disease']=='COVID-19'].groupby('date').size().reset_index()
n_news1.columns=['date', 'n_news1']
data = data.merge(n_news1, on='date', how='outer')

# Vrial 기사 합계
n_news2 = news[news['group']=='Viral'].groupby('date').size().reset_index()
n_news2.columns=['date', 'n_news2']
data = data.merge(n_news2, on='date', how='outer')
data
##################################################################
# 5월 5일까지 제공된 데이터에 한하여 모든 feature들이 14일 후를 나타내도록 변환. date를 14일 fore-shift
data['date'] = data['date'].apply(lambda x:x+timedelta(days=14))
##################################################################

# 실제 당일의 유입 확진자(정답 데이터) 병합
label = pd.read_csv(PATH_CONFIRMED).drop(columns=['total'])
label.columns = ['date', 'China', 'AS', 'EU', 'AM', 'AF', 'OC']
label['date'] = pd.to_datetime(label['date'], format='%Y-%m-%d')
label_ = pd.melt(label, id_vars='date', var_name='continent', value_vars=label.columns[1:], value_name='y')
data = data.merge(label_, on=['date', 'continent'], how='outer').sort_values(by='date')

# 질병관리본부 데이터 기준, 20년 3월 22일 이후의 데이터만 사용
data = data[data['date'] >= datetime.strptime('2020-03-22', '%Y-%m-%d')]

# 중국은 제공되는 부가 데이터셋의 정보폭이 좁아 제외
data = data[data['continent'] != 'China']

# 간헐적 미수집 데이터(NaN)는 앞, 뒤 레코드의 평균으로 대체(제외 시 sample instances 부족)
numerics = ['n_roamer', 'd_stay', 'confirmed', 'deaths', 'tests']
data[numerics] = data.groupby('continent').apply(lambda x: (x[numerics].fillna(method='ffill')+
                                                            x[numerics].fillna(method='bfill'))/2)
data
# 입국 국제항공편 스케줄 데이터(IATA code)
# 14일 뒤의 스케줄도 제공되므로 실제 예측날짜와 매칭
iata = pd.read_csv(PATH_IATA)
air = pd.read_csv(PATH_AIRPORT_SCHEDULE)

iata['continent'] = iata.apply(lambda x:'AM' if x['iso_country'] in ['US', 'CA', 'MX'] else x['continent'], axis=1)
ttable = iata[['iata_code', 'continent']].dropna()
ttable['continent'].replace('SA', 'AM', inplace=True)
ttable['continent'].replace('NA', 'AM', inplace=True)
ttable.set_index('iata_code', inplace=True)
air['arrival'] = air['arrival'].apply(lambda x:ttable.loc[x].values[0])
air['arrival'] = air['arrival'].apply(lambda x:x[0] if len(x)==1 else x)
air.rename(columns={'arrival':'continent'}, inplace=True)
air = air.groupby(['date', 'continent']).size().reset_index()
air.columns = ['date', 'continent', 'enter']
air['date'] = pd.to_datetime(air['date'], format='%Y-%m-%d')

# n번째 날의 유입사례 보도자료는 대개 n-1번째 날의 입국항공 스케줄을 따르므로 날짜마다 day(+1)
air['date'] = air['date'].apply(lambda x:x+timedelta(days=1))
data = data.merge(air, on=['date', 'continent'], how='left')

data
# feature engineering
def corona_fe(df, use_y_lags):
    # 요일
    df['week'] = df['date'].apply(lambda x:x.strftime('%a'))

    # lag features
    if use_y_lags:
        for d in range(1, 8):
            df[f'y_lag{d}'] = df.groupby(['continent'])['y'].shift(d)
    for d in range(1, 8):
        df[f'enter_lag{d}'] = df.groupby(['continent'])['enter'].shift(d)
    for info in daily_fname.keys():
        for d in range(2, 8):
            df[f'{info}_mean{d}'] = df.groupby(['continent'])[info].rolling(d, min_periods=1).mean().reset_index(0, drop=True)
            
    # rolling mean features(time-window)
    for d in range(2, 8):
        if use_y_lags:
            df[f'y_mean{d}'] = df.groupby(['continent'])['y_lag1'].rolling(d, min_periods=1).mean().reset_index(0, drop=True)
        df[f'enter_mean{d}'] = df.groupby(['continent'])['enter'].rolling(d, min_periods=1).mean().reset_index(0, drop=True)
    for d in range(2, 8):
        df[f'n_news1_mean{d}'] = df.groupby(['continent'])['n_news1'].rolling(d, min_periods=1).mean().reset_index(0, drop=True)
        df[f'n_news2_mean{d}'] = df.groupby(['continent'])['n_news2'].rolling(d, min_periods=1).mean().reset_index(0, drop=True)
    for d in range(2, 8):
        df[f'stay_mean{d}'] = df.groupby(['continent'])['d_stay'].rolling(d, min_periods=1).mean().reset_index(0, drop=True)
        df[f'roamer_mean{d}'] = df.groupby(['continent'])['n_roamer'].rolling(d, min_periods=1).mean().reset_index(0, drop=True)   
               
    # categorical features --> one hot encoded features
    df = pd.concat([df, pd.get_dummies(df['continent'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['week'])], axis=1)
    
    df = df.drop(columns=['continent', 'week', 'd_stay', 'n_roamer', 'confirmed', 'tests', 'deaths', 'n_news1', 'n_news2'])
    return df
print('----- training - prediction ------')
train_data = corona_fe(data.copy(), False)
test_data = train_data[train_data['date'] >= datetime.strptime('2020-05-06', '%Y-%m-%d')]
train_data = train_data[train_data['date']<datetime.strptime('2020-05-06', '%Y-%m-%d')].dropna()
test_data = test_data.drop(columns=['y']).dropna()

X_train = train_data.drop(columns=['y', 'date'])
y_train = train_data['y']
X_test = test_data.drop(columns=['date'])

# 회귀모델 정의
model = xgb.XGBRegressor(objective='reg:squarederror',
                        eval_metric='rmse')
# hyper marameters
dict_params = {
          'eta':0.005,
          'colsample_bytree':0.111,
          'max_depth':4,
          'min_child_weight':9,
          'gamma': 0,
          'reg:alpha' : 0,
          'n_estimators':1024,
          'n_rounds': 40,
          'subsample': 1.0
         }
model.set_params(**dict_params)

# 학습-예측
model.fit(X_train, y_train)
pred = model.predict(X_test)
test_data['pred'] = pred
results = test_data.groupby('date')['pred'].sum()
display(results)

# 5월 10일 기준 공시된 유입 확진자 수와 비교
actual_confirmed = [2, 3, 11, 1, 8]
rmse = np.sqrt(mean_squared_error(actual_confirmed, results[:5]))
acc = 1 - (np.sum(np.square(actual_confirmed - results[:5])) / np.sum(np.square(actual_confirmed)))
print(f'results comparison until May 10th')
print(f'rmse : {rmse}')
print(f'acc : {acc}')
plt.figure(figsize=(20, 20))
pd.DataFrame(data=model.feature_importances_, index=X_train.columns, columns=['importance'])\
            .sort_values(by='importance', ascending=True).plot.barh(figsize=(20,20))
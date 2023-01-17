import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import time

from datetime import datetime

from scipy import integrate, optimize

import ipywidgets as widgets

from IPython.display import display, clear_output

import warnings

warnings.filterwarnings('ignore')



# ML libraries

import lightgbm as lgb

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.metrics import mean_squared_error
# Read data

confirms_data = pd.read_csv("/kaggle/input/time-series-data/confirm.csv")

deaths_data = pd.read_csv("/kaggle/input/time-series-data/death.csv")

recovered_data = pd.read_csv("/kaggle/input/time-series-data/recovered.csv")

# Get columns name

column_comfirms_names= list(confirms_data.columns.values)

column_deaths_names= list(deaths_data.columns.values)

column_recovered_names= list(recovered_data.columns.values)

# Total rows and columns each tabale

print('Confirms data has total:',len(confirms_data) ,'rows, and',len(column_comfirms_names),'columns.')

print('Deaths data has total:',len(deaths_data) ,'rows, and',len(column_deaths_names),'columns.')

print('Recovered data has total:',len(recovered_data) ,'rows, and',len(column_recovered_names),'columns.')



# Number of countries in data

print("Number of Country/Region: ", confirms_data['Country/Region'].nunique())

print("From day", column_deaths_names[4], "to day", column_deaths_names[-1], ":", len(column_comfirms_names)-4, "days")
display(confirms_data)

display(deaths_data)

display(recovered_data)
confirm_data_totals = confirms_data.sum(axis = 0, skipna = True)[3:]

deaths_data_totals = deaths_data.sum(axis = 0, skipna = True)[3:]

recovered_data_totals = recovered_data.sum(axis = 0, skipna = True)[3:]

total = [confirm_data_totals,deaths_data_totals,recovered_data_totals]

totals = pd.concat(total,axis=1)

totals.columns = [ 'Confirmed','Deaths','Recovered']



fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(17,7))

totals.plot(ax=ax1)

ax1.set_title("Global cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)



deaths_data_totals.plot(ax=ax2, color='orange')

ax2.set_title("Global deaths cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)



recovered_data_totals.plot(ax=ax3, color='green')

ax3.set_title("Global recovered cases", size=13)

ax3.set_ylabel("Number of cases", size=13)

ax3.set_xlabel("Date", size=13)
def showPLTByCountry(country,axs):

    confirm_data_totals= confirms_data.loc[confirms_data["Country/Region"] == country].sum(axis = 0, skipna = True)[4:]

    deaths_data_totals = deaths_data.loc[deaths_data["Country/Region"] == country].sum(axis = 0, skipna = True)[4:]

    recovered_data_totals = recovered_data.loc[recovered_data["Country/Region"] == country].sum(axis = 0, skipna = True)[4:]

    total_ = [confirm_data_totals,deaths_data_totals,recovered_data_totals]

    total = pd.concat(total_,axis=1)

    total.columns = [ 'Confirmed','Deaths','Recovered']

    total.plot(ax=axs)

    axs.set_title(country+"'s cases", size=13)

    axs.set_ylabel("Number of cases", size=13)

#     axs.set_xlabel("Date", size=13)

fig1, (axs) = plt.subplots(2,3,figsize=(17,7))

showPLTByCountry("US",axs[0][0])

showPLTByCountry("Brazil",axs[0][1])

showPLTByCountry("India",axs[0][2])

showPLTByCountry("Russia",axs[1][0])

showPLTByCountry("South Africa",axs[1][1])

showPLTByCountry("China",axs[1][2])

# susceptible

def dS_dt(S, I, beta):

    return -beta*S*I

    

# infected

def dI_dt(S, I, beta, gamma):

    return beta*S*I - gamma*I



# recovered

def dR_dt(I, gamma):

    return gamma*I
def runge_kutta(Sn, In, Rn, beta, gamma, h):

    ks1 = dS_dt(Sn, In, beta)

    ki1 = dI_dt(Sn, In, beta, gamma)

    kr1 = dR_dt(In, gamma)

    

    ks2 = dS_dt(Sn + 0.5*h*ks1, In + 0.5*h*ki1, beta)

    ki2 = dI_dt(Sn + 0.5*h*ks1, In + 0.5*h*ki1, beta, gamma)

    kr2 = dR_dt(In + 0.5*h*ki1, gamma)

    

    ks3 = dS_dt(Sn + 0.5*h*ks2, In + 0.5*h*ki2, beta)

    ki3 = dI_dt(Sn + 0.5*h*ks2, In + 0.5*h*ki2, beta, gamma)

    kr3 = dR_dt(In + 0.5*h*ki2, gamma)

    

    ks4 = dS_dt(Sn + h*ks3, In + h*ki3, beta)

    ki4 = dI_dt(Sn + h*ks3, In + h*ki3, beta, gamma)

    kr4 = dR_dt(In + h*ki3, gamma)

    

    Sn_1 = Sn + (ks1 + 2*ks2 + 2*ks3 + ks4)*h/6

    In_1 = In + (ki1 + 2*ki2 + 2*ki3 + ki4)*h/6

    Rn_1 = Rn + (kr1 + 2*kr2 + 2*kr3 + kr4)*h/6

    

    return Sn_1, In_1, Rn_1

    
def SIR_model(N, beta, gamma, h):

    # N là dân số thế giới hiện tại

    # giả sử ban đầu có 1 người nhiễm nên S0=N-1, I0=1, R0=0

    # ở đây ta sẽ chuẩn hóa dữ liệu nẳm trong [0,1]

    s = float(N-1)/N

    i = float(1)/N

    r = 0.

    

    susceptible, infected, recovered = [], [], []

    #ta sẽ lặp 10000 lần (time-steps) để lấy dữ liệu tương ứng

    for k in range(10000):

        susceptible.append(s)

        infected.append(i)

        recovered.append(r)

        s, i, r = runge_kutta(s, i, r, beta, gamma, h)

        

    return susceptible, infected, recovered
N = 7800000000 # dân số thế giới hiện tại

beta = 0.7

gamma = 0.2

h = 0.1



susceptible, infected, recovered = SIR_model(N, beta, gamma, h)



f = plt.figure(figsize=(8,5)) 

plt.plot(susceptible, '#2ca02c', label='susceptible');

plt.plot(infected, '#ff7f0e', label='infected');

plt.plot(recovered, '#17becf', label='recovered/deceased');

plt.title("SIR model")

plt.xlabel("time", fontsize=10);

plt.ylabel("Normalized population", fontsize=10);

plt.legend(loc='best')

plt.xlim(0,1000)

plt.savefig('SIR_model.png')

plt.show()
t= confirms_data.loc[confirms_data["Country/Region"] == "US"]

column_day= list(t.columns.values)

column_day= column_day[4:]

t = t.sum(axis = 0, skipna = True).to_frame()

t = t.T



x= t.loc[:,column_day[0]:column_day[-1]]

x = x.diff(axis=1).fillna(0)

x.values[0][0] = t[column_day[0]].values[0]



population = float(330578810)

# population = float(1439323776)



day_count = list(range(1,len(column_day)+1))

xdata = day_count

ydata = np.array(x.values[0], dtype=float)

xdata = np.array(xdata, dtype=float)



N = population

inf0 = ydata[0]

sus0 = N - inf0

rec0 = 0.0



def sir_model(y, x, beta, gamma):

    sus = -beta * y[0] * y[1] / N

    rec = gamma * y[1]

    inf = -(sus + rec)

    return sus, inf, rec



def fit_odeint(x, beta, gamma):

    return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]



popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)

fitted = fit_odeint(xdata, *popt)



plt.plot(xdata, ydata, 'o')

plt.plot(xdata, fitted)

plt.title("Fit of SIR model for US confirmed cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])

t = confirms_data.sum(axis = 0, skipna = True).to_frame()

t = t.T

column_day= list(t.columns.values)

column_day= column_day[4:]



x= t.loc[:,column_day[0]:column_day[-1]]

x = x.diff(axis=1).fillna(0)

x.values[0][0] = t[column_day[0]].values[0]



population = float(7800000000)

# population = float(1439323776)



day_count = list(range(1,len(column_day)+1))

xdata = day_count

ydata = np.array(x.values[0], dtype=float)

xdata = np.array(xdata, dtype=float)



N = population

inf0 = ydata[0]

sus0 = N - inf0

rec0 = 0.0



def sir_model(y, x, beta, gamma):

    sus = -beta * y[0] * y[1] / N

    rec = gamma * y[1]

    inf = -(sus + rec)

    return sus, inf, rec



def fit_odeint(x, beta, gamma):

    return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]



popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)

fitted = fit_odeint(xdata, *popt)



plt.plot(xdata, ydata, 'o')

plt.plot(xdata, fitted)

plt.title("Fit of SIR model for global confirmed cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

train.Province_State.fillna("None", inplace=True)
# Do dữ liệu được lấy từ cuộc thi nên ta gộp 2 bộ dữ liệu train và test với nhau

dates_overlap = ['2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04', '2020-04-05', '2020-04-06', '2020-04-07', '2020-04-08',

                 '2020-04-09', '2020-04-10', '2020-04-11', '2020-04-12', '2020-04-13', '2020-04-14']

train2 = train.loc[~train['Date'].isin(dates_overlap)]

all_data = pd.concat([train2, test], axis = 0, sort=False)



# Chắc chắn chỉ có thông tin về số ca nhiễm và số ca tử vong trước ngày 11/03/2020

# Mục đích: ta huấn luyện với bộ dữ liệu trước ngày 11/03/2020 để dự đoán 

all_data.loc[all_data['Date'] >= '2020-04-01', 'ConfirmedCases'] = 0

all_data.loc[all_data['Date'] >= '2020-04-01', 'Fatalities'] = 0

all_data['Date'] = pd.to_datetime(all_data['Date'])



# Tạo các cột dữ liệu chứ thời gian: ngày, tháng, năm

le = preprocessing.LabelEncoder()

all_data['Day_num'] = le.fit_transform(all_data.Date)

all_data['Day'] = all_data['Date'].dt.day

all_data['Month'] = all_data['Date'].dt.month

all_data['Year'] = all_data['Date'].dt.year



# Xử lý dữ liệu bị mất trong dữ liệu bằng cách thay thế

all_data['Province_State'].fillna("None", inplace=True)

all_data['ConfirmedCases'].fillna(0, inplace=True)

all_data['Fatalities'].fillna(0, inplace=True)

all_data['Id'].fillna(-1, inplace=True)

all_data['ForecastId'].fillna(-1, inplace=True)



display(all_data.head())
all_data.info() # Số lượng phần tử non-null trong từng cột đểu bằng số lượng phần tử
# Định nghĩa các hàm tính lags và trends của dữ liệu 



def calculate_lag(dataframe, lag_list, column):

    for lag in lag_list:

        column_lag = f"{column}_{lag}"

        dataframe[column_lag] = dataframe.groupby(['Country_Region', 'Province_State'])[column].shift(lag, fill_value=0)

    return dataframe



def calculate_trend(dataframe, lag_list, column):

    df_groupby = dataframe.groupby(['Country_Region', 'Province_State'])

    for lag in lag_list:

        trend_column_lag = f"Trend_{column}_{lag}"

        dataframe[trend_column_lag] = (df_groupby[column].shift(0, fill_value=0) - 

                                df_groupby[column].shift(lag, fill_value=0))/df_groupby[column].shift(lag, fill_value=0.001)

    return dataframe
all_data = calculate_lag(all_data.reset_index(), range(1,7), 'ConfirmedCases')

all_data = calculate_lag(all_data, range(1,7), 'Fatalities')

all_data = calculate_trend(all_data, range(1,7), 'ConfirmedCases')

all_data = calculate_trend(all_data, range(1,7), 'Fatalities')

all_data.replace([np.inf, -np.inf], 0, inplace=True)

all_data.fillna(0, inplace=True)
# Đọc dữ liệu từ data file

world_population = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")



# Chọn các cột mong muốn , và sửa đổi lại tên 

world_population = world_population[['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]

world_population.columns = ['Country_Dependency', 'Population', 'Density', 'Land Area', 'Med Age', 'Urban Pop']



# Thay tên "United States" thành "US"

world_population.loc[world_population['Country_Dependency']=='United States', 'Country_Dependency)'] = 'US'



# Xóa dấu % ở cột dữ liệu Urban Pop

world_population['Urban Pop'] = world_population['Urban Pop'].str.rstrip('%')



# Replace Urban Pop and Med Age "N.A" by their respective modes

# Thay thế các giá trị "N.A" trong cột "Urban Pop" và cột "Med Age" bằng mode của nó

world_population.loc[world_population['Urban Pop']=='N.A.', 'Urban Pop'] = int(world_population.loc[world_population['Urban Pop']!='N.A.', 'Urban Pop'].mode()[0])

world_population['Urban Pop'] = world_population['Urban Pop'].astype('int16')



world_population.loc[world_population['Med Age']=='N.A.', 'Med Age'] = int(world_population.loc[world_population['Med Age']!='N.A.', 'Med Age'].mode()[0])

world_population['Med Age'] = world_population['Med Age'].astype('int16')





# Now join the dataset to our previous DataFrame and clean missings (not match in left join)- label encode cities

print("Joined dataset")

all_data = all_data.merge(world_population, left_on='Country_Region', right_on='Country_Dependency', how='left')

all_data[['Population', 'Density', 'Land Area', 'Med Age', 'Urban Pop']] = all_data[['Population', 'Density', 'Land Area', 'Med Age', 'Urban Pop']].fillna(0)

display(all_data.head())



print("Encoded dataset")

# Label encode countries and provinces. Save dictionary for exploration purposes

all_data.drop('Country_Dependency)', inplace=True, axis=1)



all_data['Country_Region'] = le.fit_transform(all_data['Country_Region'])

number_c = all_data['Country_Region']

countries = le.inverse_transform(all_data['Country_Region'])

country_dict = dict(zip(countries, number_c)) 



all_data['Province_State'] = le.fit_transform(all_data['Province_State'])

number_p = all_data['Province_State']

province = le.inverse_transform(all_data['Province_State'])

province_dict = dict(zip(province, number_p)) 

display(all_data.head())
# Trực quan dữ liệu cả hai trường hợp đối với Tây Ban Nha với dữ liệu 10 ngày cuối mà có thông tin, bắt đầu từ 01/03/2020



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



condition = (all_data['Country_Region']==country_dict['Spain']) & (all_data['Day_num']>39) & (all_data['Day_num']<=49)



# Day_num = 38 is March 1st

y1 = all_data[condition][['ConfirmedCases']]

x1 = range(0, len(y1))

ax1.plot(x1, y1, 'bo--')

ax1.set_title("Spain ConfirmedCases between days 39 and 49")

ax1.set_xlabel("Days")

ax1.set_ylabel("ConfirmedCases")





y2 = all_data[condition][['ConfirmedCases']].apply(lambda x: np.log(x))

x2 = range(0, len(y2))

ax2.plot(x2, y2, 'bo--')

ax2.set_title("Spain Log ConfirmedCases between days 39 and 49")

ax2.set_xlabel("Days")

ax2.set_ylabel("Log ConfirmedCases")
# Chọn lọc các đặc trưng làm đầu vào mô hình

data = all_data.copy()

features = ['Id', 'ForecastId', 'Country_Region', 'Province_State', 'ConfirmedCases', 'Fatalities', 

       'Day_num']

data = data[features]



# Áp dụng biến đổi Logarithm cho ConfirmedCases và Fatalities cột

data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].astype('float64')

data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log1p(x))



# Replace infinites

data.replace([np.inf, -np.inf], 0, inplace=True)
# Chia thành tập huấn luyện và tập kiểm thử

def split_data(df, train_lim, test_lim):

    

    df.loc[df['Day_num']<=train_lim , 'ForecastId'] = -1

    df = df[df['Day_num']<=test_lim]

    

    # Tập huấn luyện

    x_train = df[df.ForecastId == -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)

    y_train_1 = df[df.ForecastId == -1]['ConfirmedCases']

    y_train_2 = df[df.ForecastId == -1]['Fatalities']



    # Tập kiểm thử 

    x_test = df[df.ForecastId != -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)



    # Loại bỏ hai cột Id và ForecastId

    x_train.drop('Id', inplace=True, errors='ignore', axis=1)

    x_train.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    x_test.drop('Id', inplace=True, errors='ignore', axis=1)

    x_test.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    

    return x_train, y_train_1, y_train_2, x_test
# hàm mô hình hồi quy tuyến tính

def lin_reg(X_train, Y_train, X_test):

    regr = linear_model.LinearRegression()



    # Huấn luyện mô hình dựa trên tập huấn luyện

    regr.fit(X_train, Y_train)



    # Dự đoán tập kiểm thử

    y_pred = regr.predict(X_test)

    

    return regr, y_pred
from datetime import timedelta, date



def daterange(start_date, end_date):

    for n in range(int((end_date - start_date).days)):

        yield start_date + timedelta(n)



start_date = date(2020, 3, 1)

end_date = date(2020, 4, 15)



dates_list = [single_date.strftime("%Y-%m-%d") for single_date in daterange(start_date, end_date)]
def plot_linreg_basic_country(data, country_name, dates_list, day_start, shift, train_lim, test_lim):

    

    data_country = data[data['Country_Region']==country_dict[country_name]]

    data_country = data_country.loc[data_country['Day_num']>day_start]

    X_train, Y_train_1, Y_train_2, X_test = split_data(data_country, train_lim, test_lim)

    model, pred = lin_reg(X_train, Y_train_1, X_test)



    # Create a df with both real cases and predictions (predictions starting on March 12th)

    X_train_check = X_train.copy()

    X_train_check['Target'] = Y_train_1



    X_test_check = X_test.copy()

    X_test_check['Target'] = pred



    X_final_check = pd.concat([X_train_check, X_test_check])



    # Select predictions from March 1st to March 25th

    predicted_data = X_final_check.loc[(X_final_check['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].Target

    real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))]['ConfirmedCases']

    dates_list_num = list(range(0,len(dates_list)))



    # Plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

    ax1.plot(list(range(len(predicted_data))), np.expm1(predicted_data))

    ax1.plot(list(range(len(real_data))), real_data)

    ax1.axvline(30-shift, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_xlabel(f"Day count (from March {1+shift})")

    ax1.set_ylabel("Confirmed Cases")



    ax2.plot(list(range(len(predicted_data))), predicted_data)

    ax2.plot(list(range(len(real_data))), np.log1p(real_data))

    ax2.axvline(30-shift, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax2.set_xlabel(f"Day count (from March {str(1+shift)})")

    ax2.set_ylabel("Log Confirmed Cases")



    plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))
train_lim, test_lim = 69, 112



output_lr = widgets.Output()



def linear_reg_country():

    country_name = country_combobox.value

    march_day = marchDay_text.value

    if not (country_name and march_day):

        return

    march_day = int(march_day)

    day_start = 39 + march_day # 39 là thứ tự của ngày 01/03/2020

    date_list_temp = dates_list[march_day:]

    with output_lr:

        output_lr.clear_output()

        plot_linreg_basic_country(data, country_name, date_list_temp, day_start, march_day, train_lim, test_lim)

        plt.show()

        

country_combobox = widgets.Combobox(

    placeholder='Country',

    options= tuple(country_dict.keys()),

    description='Country: ',

    ensure_option=True,

    disabled=False

)    



marchDay_text = widgets.Text(

    placeholder='Enter number',

    description="March day:"

)



submit_lr_btn = widgets.Button(description="run")



def run_linear_reg_country(b):

    linear_reg_country()



display(country_combobox, marchDay_text, submit_lr_btn, output_lr)



submit_lr_btn.on_click(run_linear_reg_country)
# Hàm chia tập huấn luyện và tập test cho dự đoán sau 1 ngày

def split_data_one_day(df, d, train_lim, test_lim):

    df.loc[df['Day_num']<=train_lim , 'ForecastId'] = -1

    df = df[df['Day_num']<=test_lim]

    

    # Tập huấn luyện

    x_train = df[df.Day_num<d]

    y_train_1 = x_train.ConfirmedCases

    y_train_2 = x_train.Fatalities

    x_train.drop(['ConfirmedCases', 'Fatalities'], axis=1, inplace=True)

    

    # Tập kiểm thử 

    x_test = df[df.Day_num==d]

    x_test.drop(['ConfirmedCases', 'Fatalities'], axis=1, inplace=True)

    

    # Loại bỏ hai cột Id và ForcastId 

    x_train.drop('Id', inplace=True, errors='ignore', axis=1)

    x_train.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    x_test.drop('Id', inplace=True, errors='ignore', axis=1)

    x_test.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    

    return x_train, y_train_1, y_train_2, x_test





# Hàm cấu hình biểu đồ 

def config_axis(axis, march_day, msg):

    axis.axvline(30-march_day, linewidth=2, ls = ':', color='grey', alpha=0.5)

    axis.set_xlabel("Day count (starting on March " + str(march_day) + "))")

    axis.set_ylabel(msg)

    return axis





# Hàm chuẩn bị dữ liệu để vẽ

def prepare_data(data, train, country_name, day_start, dates_list, fatalities=False):

    column = "ConfirmedCases"

    if fatalities:

        column = "Fatalities"

    

    predicted_data = data.loc[(data['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))][column]

    real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))][column]

    

    dates_list_num = list(range(0,len(dates_list)))

    

    return predicted_data, real_data, dates_list_num





def plot_real_vs_prediction_country(data, train, country_name, day_start, dates_list, march_day):

    predicted_data, real_data, dates_list_num = prepare_data(data, train, country_name, day_start, dates_list)



    # Plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

    

    config_axis(ax1, march_day, "Confirmed Cases")

    ax1.plot(list(range(len(predicted_data))), np.expm1(predicted_data))

    ax1.plot(list(range(len(real_data))), real_data)

    

    config_axis(ax2, march_day, "Log Confirmed Cases")

    ax2.plot(dates_list_num, predicted_data)

    ax2.plot(dates_list_num, np.log1p(real_data))



    plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))

    

    

def plot_real_vs_prediction_country_fatalities(data, train, country_name, day_start, dates_list, march_day):

    predicted_data, real_data, dates_list_num = prepare_data(data, train, country_name, day_start, dates_list, True)



    # Plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

    

    ax1 = config_axis(ax1, march_day, "Fatalities Cases")

    ax1.plot(dates_list_num, np.expm1(predicted_data))

    ax1.plot(dates_list_num, real_data)

    

    ax2 = config_axis(ax2, march_day, "Log Fatalities Cases")

    ax2.plot(dates_list_num, predicted_data)

    ax2.plot(dates_list_num, np.log1p(real_data))

    plt.suptitle(("Fatalities predictions based on Log-Lineal Regression for "+country_name))
# Hàm dự đoán sử dụng mô hình Hồi quy tuyến tính với thêm đặc trưng lags cho một quốc gia

def lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict, train_lim, test_lim):

    data = all_data.copy()

    features = ['Id', 'Province_State', 'Country_Region',

           'ConfirmedCases', 'Fatalities', 'ForecastId', 'Day_num']

    data = data[features]



    # Select country an data start (all days)

    data = data[data['Country_Region']==country_dict[country_name]]

    data = data.loc[data['Day_num'] > day_start]



    # Lags

    data = calculate_lag(data, range(1,lag_size), 'ConfirmedCases')

    data = calculate_lag(data, range(1,8), 'Fatalities')

    

    # Chọn ra các cột thuộc tính Confirmed

    filter_col_confirmed = [col for col in data if col.startswith('Confirmed')]

    filter_col_fatalities= [col for col in data if col.startswith('Fataliti')]

    filter_col = np.append(filter_col_confirmed, filter_col_fatalities)

    

    # Apply log transformation

    data[filter_col] = data[filter_col].apply(lambda x: np.log1p(x))

    data.replace([np.inf, -np.inf], 0, inplace=True)

    data.fillna(0, inplace=True)





    # Start/end of forecast

    start_fcst = all_data[all_data['Id']==-1].Day_num.min()

    end_fcst = all_data[all_data['Id']==-1].Day_num.max()



    for d in list(range(start_fcst, end_fcst+1)):

        X_train, Y_train_1, Y_train_2, X_test = split_data_one_day(data, d, train_lim, test_lim)

        model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

        data.loc[(data['Country_Region']==country_dict[country_name]) 

                 & (data['Day_num']==d), 'ConfirmedCases'] = pred_1[0]

        model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

        data.loc[(data['Country_Region']==country_dict[country_name]) 

                 & (data['Day_num']==d), 'Fatalities'] = pred_2[0]



        # Tính toán lại lags

        data = calculate_lag(data, range(1,lag_size), 'ConfirmedCases')

        data = calculate_lag(data, range(1,8), 'Fatalities')

        data.replace([np.inf, -np.inf], 0, inplace=True)

        data.fillna(0, inplace=True)

   

    return data
train_lim, test_lim = 69, 112



output = widgets.Output()



def linear_reg_with_lag_country():

    country_name = country_combobox_lag.value

    march_day = marchDay_text_lag.value

    lag_size = lagSize_text.value

    

    if not (country_name and march_day and lag_size):

        return

    march_day = int(march_day)

    lag_size = int(lag_size)

    day_start = 39 + march_day # 39 là thứ tự của ngày 01/03/2020

    date_list_temp = dates_list[march_day:]

    data_c = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict, train_lim, test_lim)

    with output:

        output.clear_output()

        print(country_name, march_day, lag_size)

        plot_real_vs_prediction_country(data_c, train, country_name, day_start, date_list_temp, march_day)

        plot_real_vs_prediction_country_fatalities(data_c, train, country_name, day_start, date_list_temp, march_day)

        plt.show()



country_combobox_lag = widgets.Combobox(

    placeholder='Country',

    options= tuple(country_dict.keys()),

    description='Country: ',

    ensure_option=True,

    disabled=False

)    



marchDay_text_lag = widgets.Text(

    placeholder='Enter number',

    description="March day:"

)

lagSize_text = widgets.Text(

    placeholder="Enter number",

    description="Lag size:"

)



submit_btn = widgets.Button(description="run")



def run_linear_reg_with_lag_country(button):

    linear_reg_with_lag_country()

    

display(country_combobox_lag, marchDay_text_lag, lagSize_text, submit_btn, output)



submit_btn.on_click(run_linear_reg_with_lag_country)



# Spain, Italy, Germany, Albania, Andora
# Định nghĩa hàm Logistic tổng quát

def logistic_function(x, a, b, c, d):

    return a / (1. + np.exp(-c * (x - d))) + b



# Huấn luyện mô hình tìm ra bộ tham số tối ưu nhất

def fit_logistic(all_data, country_name, province_name, train_lim, target):

    data_cp = all_data.loc[(all_data['Country_Region']==country_dict[country_name]) & (all_data['Province_State']==province_dict[province_name])]

    y = data_cp.loc[(data_cp['Day_num'])<=train_lim, target].astype(np.int32)

    x = list(range(0, len(y)))



    # Khởi tạo bộ tham số đầu tiên

    p0 = [0,1,1,0]



    (a_, b_, c_, d_), cov = optimize.curve_fit(logistic_function, x, y, bounds=(0, [500000., 10., 1000., 1000., ]), p0=p0, maxfev=10**9)

    y_fit = logistic_function(x, a_, b_, c_, d_)

    return x, y, y_fit, (a_, b_, c_, d_), cov



# Vẽ hàm logistic

def plot_logistic(x, y, y_fit, country_name, province_name, target):

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(x, y, 'o')

    ax.plot(x, y_fit, '-')

    ax.set_xlabel("Day count (starting on January 22nd)")

    ax.set_ylabel(target)

    ax.set_title("Fit to logistic regression for "+ country_name+"/"+province_name)

    

# Vẽ giá trị dự đoán cho một quốc gia

def plot_logistic_country(all_data, train, dates_overlap, country_name, province_name, valid_num, target, x, a_, b_, c_, d_):

    forecast = logistic_function(list(range(len(x)+60)), a_, b_, c_, d_)

    df_train = train.loc[(train['Country_Region']==country_name) & (train['Province_State']==province_name), target]

    df_fcst = forecast[:len(df_train)]

    dates = list(range(len(df_train)))

    

    # Vẽ kết quả

    fig, (ax1) = plt.subplots(1, 1, figsize=(6,4))

    ax1.plot(dates, df_fcst)

    ax1.plot(dates, df_train)

    ax1.axvline(len(df_train)-valid_num-1, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax1.set_title("Actual ConfirmedCases vs predictions based on Logistic curve for "+country_name + "/"+province_name)

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_xlabel("Day count starting on January 22nd")

    ax1.set_ylabel("ConfirmedCases")





train_lim = 69

valid_lim = 84 

test_lim = 112

valid_num=valid_lim-train_lim 
country_name = 'Spain'

province_name = 'None'



x, y, y_fit, (a_, b_, c_, d_), cov = fit_logistic(all_data, country_name, province_name, train_lim, 'ConfirmedCases')

plot_logistic(x, y, y_fit, country_name, province_name, 'ConfirmedCases')

plot_logistic_country(all_data, train, dates_overlap, country_name, province_name, valid_num, 'ConfirmedCases', x, a_, b_, c_, d_)
country_name = 'Italy'

province_name = 'None'

x, y, y_fit, (a_, b_, c_, d_), cov = fit_logistic(all_data, country_name, province_name, train_lim, 'ConfirmedCases')

plot_logistic(x, y, y_fit, country_name, province_name, 'ConfirmedCases')

plot_logistic_country(all_data, train, dates_overlap, country_name, province_name, valid_num, 'ConfirmedCases', x, a_, b_, c_, d_)
country_name = 'Germany'

province_name = 'None'

x, y, y_fit, (a_, b_, c_, d_), cov = fit_logistic(all_data, country_name, province_name, train_lim, 'ConfirmedCases')

plot_logistic(x, y, y_fit, country_name, province_name, 'ConfirmedCases')

plot_logistic_country(all_data, train, dates_overlap, country_name, province_name, valid_num, 'ConfirmedCases', x, a_, b_, c_, d_)
country_name = 'Andorra'

province_name = 'None'

x, y, y_fit, (a_, b_, c_, d_), cov = fit_logistic(all_data, country_name, province_name, train_lim, 'ConfirmedCases')

plot_logistic(x, y, y_fit, country_name, province_name, 'ConfirmedCases')

plot_logistic_country(all_data, train, dates_overlap, country_name, province_name, valid_num, 'ConfirmedCases', x, a_, b_, c_, d_)
country_name = 'China'

province_name = 'Hubei'

x, y, y_fit, (a_, b_, c_, d_), cov = fit_logistic(all_data, country_name, province_name, train_lim, 'ConfirmedCases')

plot_logistic(x, y, y_fit, country_name, province_name, 'ConfirmedCases')

plot_logistic_country(all_data, train, dates_overlap, country_name, province_name, valid_num, 'ConfirmedCases', x, a_, b_, c_, d_)
#############

# Daily report : from 22/1/2020

data_us = pd.read_csv("/kaggle/input/datadeathbygroups/us-states.csv")

# data_us.to_excel ('hle.xlsx', index = None, header=True)

# Weekly report : from 

data_by_group = pd.read_csv("/kaggle/input/datadeathbygroups/death_weekly.csv")

##############



# Connecticut

# Daily report in Connecticut : Chi dung duoc data nay

data_by_group_st_conn = pd.read_csv("/kaggle/input/datadeathbygroups/case_connecticut.csv")

# Sort the data frame by date updated

# data_by_group_st_conn["DateUpdated"] = pd.to_datetime(data_by_group_st_conn["DateUpdated"])

print ('Total case infected in 08/05/2020 (mm/dd/yyyy):' ,sum(data_by_group_st_conn.loc[data_by_group_st_conn["DateUpdated"] == "08/05/2020"]['Total cases'].tolist()))

display(data_by_group_st_conn)



print(data_by_group_st_conn['AgeGroups'].unique())
# Sắp xếp lại dữ liệu theo ngày tháng

data_by_group_st_conn = data_by_group_st_conn.sort_values(by="DateUpdated")

days= data_by_group_st_conn['DateUpdated'].unique().tolist()

print("From day", days[0], "to day", days[-1], ":",len(days) ,"days")

# Thay đổi giá trị của age groups

col_age = data_by_group_st_conn['AgeGroups'].tolist()

col_age = [i.replace(" ", "").replace("andolder","+").replace('19-Oct','10-19') for i in col_age]

data_by_group_st_conn.loc[:,'AgeGroups'] = col_age



print(data_by_group_st_conn['AgeGroups'].unique())
import statistics 

fig, (myax,myax2) = plt.subplots(1,2,figsize=(17,7))

date = data_by_group_st_conn['DateUpdated'].tolist();

case_by_age = []

col_age = data_by_group_st_conn['AgeGroups'].unique().tolist()

col_age.sort()

for i in col_age:

    newlist= data_by_group_st_conn.loc[data_by_group_st_conn["AgeGroups"] == i]['Total cases'].tolist()

    case_by_age.append(newlist)

idx =0

for lists in case_by_age:

    myax.plot(lists, label= str(col_age[idx]))

    idx+=1



# Add legend

myax.legend(loc=2, ncol=2)

strTitle= '1: Total cases infected from ' +days[0]

myax.title.set_text(strTitle)

myax.set_ylabel("Number of cases", size=13)

myax.set_xlabel("Days", size=13)

# fig, (myax) = plt.subplots(1,1,figsize=(17,7))

# date = data_by_group_st_conn['DateUpdated'].tolist();

# date.sort()

case_by_age = []

for i in col_age:

    newlist= data_by_group_st_conn.loc[data_by_group_st_conn["AgeGroups"] == i]['Total deaths'].tolist()

    case_by_age.append(newlist)

idx =0

for lists in case_by_age:

    myax2.plot(lists, label= str(col_age[idx]))

    idx+=1



# Add legend

plt.legend(loc=2, ncol=2)

strTitle= '2: Total deaths from ' +days[0]

myax2.title.set_text(strTitle)

myax2.set_ylabel("Number of cases", size=13)

myax2.set_xlabel("Days", size=13)

## Ratio for deaths and infected

fig, (ratio) = plt.subplots(1,1,figsize=(17,7))

case_by_age = []

for i in col_age:

    newlistCase= data_by_group_st_conn.loc[data_by_group_st_conn["AgeGroups"] == i]['Total cases'].tolist()

    newlistDeaths= data_by_group_st_conn.loc[data_by_group_st_conn["AgeGroups"] == i]['Total deaths'].tolist()

    newlist =[newlistDeaths[i]/newlistCase[i] for i in range(len(newlistCase))]

    print(i ,' : ', statistics.mean(newlist))



    case_by_age.append(newlist)

idx =0

for lists in case_by_age:

    ratio.plot(lists, label= str(col_age[idx]))

    idx+=1



# Add legend

ratio.legend(loc=2, ncol=2)

strTitle= '3: Ratio between deaths and infected case ' +days[0]

ratio.title.set_text(strTitle)

ratio.set_ylabel("Ratio", size=13)

ratio.set_xlabel("Days", size=13)
import numpy as np

import pandas as pd

import math

import datetime

import matplotlib.pyplot as plt
dataset= pd.read_csv('https://raw.githubusercontent.com/phamdinhkhanh/AISchool/master/data_stocks.csv?fbclid=IwAR04kY8Q_B6FFrtSDp40umzfWQHPqBENdIt9b-GywbJ9ZyT08X5nWCoByas', sep=',', index_col=0)

dataset.head()

dataset.dtypes
dataset.isna().sum()
dataset.isnull().sum()
dataset['Symbols'].unique() #Check Xem có mã nào thiếu hoàn toàn hoặc nằm ngoài 4 mã mà đề bài nhắc đến hay không

dmean= dataset.drop(['Volume', 'Adj Close'], axis=1).groupby(['Symbols']).mean() #Sắp xếp lại bảng dữ liệu và tìm mean
print ('Mức giá Open, High, Low, Close trung bình của mỗi mã chứng khoán trong thời gian tồn tại như sau:')

print ('\n')

print (dmean.round({'High':2, 'Low':2, 'Open':2, 'Close':2}))

dataset['Date']= pd.to_datetime(dataset['Date']) #Chuyển data types của cột Date trước khi sử dụng

dataset.dtypes

def fil(Sy): #Lọc ra dữ liệu của từng mã chứng khoán thành 4 bộ khác nhau

    return dataset.loc[(dataset['Symbols']==Sy), ['Close', 'Symbols','Date']].reset_index().copy()



#Vì sẽ chỉ còn làm việc với giá Close của các mã chứng khoán kể từ đây nên ta chỉ còn cần đến 'Close', 'Symbols' và 'Date'.
#t = log(Close ngày t) - log(Close ngày (t-1))



def t(c,prec): #c=Close ngày t; prec hay previous c =Close ngày (t-1)

    return math.log(c)-math.log(prec)
def exe(sym):                             #Thực thi xuất ra bảng tính toán lợi suất theo ngày.

                                          #Ở đây không quan tâm đến các ngày t7,cn (sàn GD nghỉ ko có data)

    lsol = list()

    lsoldate = list()

    for i in range(sym['Date'].count()-1):

            d=sym.Date

            cl=sym.Close

            if i > 0:               #Ở đây chỉ cần không phải là ngày đầu tiên thì các tuần về sau,

                                    #lợi suất của thứ 2 được tính liền với thứ 6 tuần trước đó (coi như không có t7, cn).

                

                check= d[i]-d[i-1]

        

                if check.days == 1:                #Với cách tính này thì trong trường hợp file dữ liệu bị thiếu ngày i 

                                                   #thì lợi suất ngày i+1 sẽ không được tính nữa và chuyển về giá trị 0 (null). 

                                                   #Nhược điểm của cách tính này là nó không phù hợp với trường hợp các 

                                                   #ngày nghỉ lễ diễn ra từ t2-t6.

                    sol = t(cl[i],cl[i-1])

                    lsol.append(sol)

                    lsoldate.append (d[i])

                    

                else:

                    continue

            else:

                continue



    table = list(zip(lsoldate,lsol))

    return pd.DataFrame(table, columns = ['Date', 'Loi Suat'])



dAAL=fil('AAL')

dAAL.head()
dAAPL=fil('AAPL')

dAAPL.head()
dAAC=fil('AAC')

dAAC.head()
dAAAU=fil('AAAU')

dAAAU.head()
lsAAL= exe(dAAL)

lsAAL.head()
lsAAC= exe(dAAC)

lsAAC.head()
lsAAAU= exe(dAAAU)

lsAAAU.head()
lsAAPL= exe(dAAPL)

lsAAPL.head()
lsAAPL.Date.dtype

lsAAPL['Date']= pd.to_datetime(lsAAPL['Date'])

lsAAC['Date']= pd.to_datetime(lsAAC['Date'])

lsAAL['Date']= pd.to_datetime(lsAAL['Date'])

lsAAAU['Date']= pd.to_datetime(lsAAAU['Date'])
import matplotlib.dates as mdates

from datetime import timedelta

def draw(lstab,ck):

    days = mdates.DayLocator()   # every days

    months = mdates.MonthLocator()  # every month

    monthsFmt = mdates.DateFormatter('%m-%Y')

    

    

    fig, ax = plt.subplots()

    ax.plot(lstab['Date'], lstab['Loi Suat'])



    # format the ticks

    ax.xaxis.set_major_locator(months)

    ax.xaxis.set_major_formatter(monthsFmt)

    ax.xaxis.set_minor_locator(days)

    

    #range setup

    dmin, dmax = ax.get_xlim()

    

    dmin=dmin - 32

    dmax=dmax + 32

    ax.set_xlim ([dmin, dmax])

    

    # format the coords message box

    def ls(x):

        return '$%1.2f' % x

    ax.format_xdata = mdates.DateFormatter('%d-%m-%Y')

    ax.format_ydata = ls

    ax.grid(True)



    # rotates and right aligns the x labels, and moves the bottom of the

    # axes up to make room for them

    fig.autofmt_xdate()

    

    plt.title('Biểu đồ lợi suất mã {}'.format(ck))    

    return plt.show()

draw(lsAAC, 'ACC')

draw(lsAAL, 'AAL')

draw(lsAAPL, 'AAPL')

draw(lsAAAU, 'AAAU')
def per(ma, ten):

    

    print ('Percentile lợi suất 25% của mã {} là: {}'. format(ten, ma.quantile (0.25) ))

    print ('Percentile lợi suất 50% của mã {} là: {}'. format(ten, ma.quantile () ))

    print ('Percentile lợi suất 75% của mã {} là: {}'. format(ten, ma.quantile (0.75) ))

    return    
per(lsAAL['Loi Suat'], 'AAL')
per(lsAAC['Loi Suat'], 'AAC')
per(lsAAPL['Loi Suat'], 'AAPL')
per(lsAAAU['Loi Suat'], 'AAAU')
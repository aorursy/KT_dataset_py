!ls ../input/to-csv-outcsv
!ls .
!ls ../input/one-year-industrial-component-degradation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl
import io
import os

#データ保存名
csv_filepath = 'to_csv_out.csv'
import datetime

# 日時データ作成 関数
def StrtoDatetime(_filename):
    filename = _filename
    #timestamp = _timestamp
    timestr = "2018-" + filename[:5] + " " + filename[6:8] + ":" +  filename[8:10] + ":" +  filename[10:12]
    dt1 = pd.to_datetime(timestr)
    #dt2 = dt1 + datetime.timedelta(seconds=timestamp)
    return dt1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

li = [];li2 = []
size = 0;size2=0

root_dir="../input/one-year-industrial-component-degradation"
i=0

for text_file in os.listdir(root_dir):
    #print(text_file)
    text_file_path = os.path.join(root_dir, text_file)
    if ('.csv' not in text_file_path):
        print(text_file)
        continue
    if ('02-29T' in text_file):
        print(text_file)
        text_file=text_file.replace('02-29T', '03-01T')
        print('>>>'+text_file)
    if ('02-30T' in text_file):
        print(text_file)
        text_file=text_file.replace('02-30T', '03-02T')
        print(">>>"+text_file)

        
    df = pd.read_csv(text_file_path,engine = "python",index_col=None, header=0)
    #df = pd.read_csv(z.open(text_file),index_col=None, header=0)
    
    df['mode'] = text_file_path[-5:-4]
    df['filename'] = text_file
    sampleNumber = text_file[-13:-10]
    df['sampleNumber'] = sampleNumber
    #df['sampletime'] = float(sampleNumber) * 10 + df['timestamp'] 
    buff=[]
    for time in df['timestamp']:
        dt1 = StrtoDatetime(text_file) + datetime.timedelta(seconds = time)
        buff.append(dt1)
    df['sampletime'] = pd.Series(buff)
    #print(type(buff))
    
    li.append(df)
    size += df.size

    
frame = pd.concat(li, axis=0, ignore_index=True)
#frame2 = pd.concat(li2, axis=0, ignore_index=True)
# データ確認
frame
# CSVファイルで保存
frame.to_csv(csv_filepath)
# ファイル確認
!ls .
!pwd
# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
# create a link to download the dataframe
create_download_link(frame,filename="to_csv_out.csv")

csv_filepath="../input/to-csv-outcsv/to_csv_out.csv"
# CSVファイルリード　※このセル以降、df_csv を使用するので破壊しないこと。
df_csv = pd.read_csv(csv_filepath, parse_dates=[0])
df_csv['sampletime'] = pd.to_datetime(df_csv['sampletime'])
df_csv = df_csv.sort_values('sampletime')
# データ確認
print("rows:" + str(len(df_csv)))
print("cals:" + str(len(df_csv.columns)))

df_csv.head(2)
df_csv
df = df_csv
# グラフ作成
COL = 4
plt.figure(figsize=(18,4))
#plt.plot(df['timestamp'], df['pCut::Motor_Torque'])
plt.plot(df['sampletime'], df.iloc[:,COL])
plt.title(df.columns.values[COL], fontsize=18)


# 時刻のフォーマットを設定
#xfmt = mpl.dates.DateFormatter('%Y-%m-%d %H:%M')
xfmt = mpl.dates.DateFormatter('%m-%d %H:%M')
plt.gca().xaxis.set_major_formatter(xfmt)

plt.show()
for num in range(2, 10):
  COL = num
  # グラフ作成
  plt.figure(figsize=(18,4))
  col="pCut::CTRL_Position_controller::Actual_speed"
  plt.plot(df['sampletime'], df.iloc[:,COL])
  #plt.plot(df['timestamp'], df.iloc[:,COL])
  plt.title(df.columns.values[COL], fontsize=18)
  
  # ロケータで刻み幅を設定
  xloc = mpl.dates.MinuteLocator(byminute=range(0,60,1))
  plt.gca().xaxis.set_major_locator(xloc)
  
  # 時刻のフォーマットを設定
  #xfmt = mpl.dates.DateFormatter('%Y-%m-%d %H:%M')
  xfmt = mpl.dates.DateFormatter('%m-%d %H:%M')
  plt.gca().xaxis.set_major_formatter(xfmt)

plt.show()
# 速度比較グラフ関数
def SpeedGraph(mode,fileno):
  data_L = df_csv.query('mode == ' + str(mode) )
  # 行選択
  SIZE = 2048 # 1ファイル行数
  SIZEMAX = int(data_L.size/data_L.columns.size)
  FILENO = fileno     
  
  ST = SIZE*FILENO
  SP = ST+SIZE-1
  dfg = data_L[ST:SP]
  
  # グラフ作成 　ブレードとフィルムの実速度比較　pSpintor::VAX_speed
  plt.figure(figsize=(15,8))
  plt.plot(dfg['sampletime'], dfg['pCut::CTRL_Position_controller::Actual_speed'],color='b', label="pCut::Actual_speed")
  plt.plot(dfg['sampletime'], dfg['pSvolFilm::CTRL_Position_controller::Actual_speed'], color='r', label="pSvolFilm::Actual_speed")
  plt.plot(dfg['sampletime'], dfg['pSpintor::VAX_speed'], color='g', label="pSpintor::VAX_speed")
  #plt.plot(dfg['timestamp'], dfg['pSpintor::VAX_speed'])
  plt.title(os.path.basename(dfg.iloc[1, 11]), fontsize=18)
  plt.legend(loc="lower left", fontsize=18) # (3)凡例表示
  
  # ロケータで刻み幅を設定　　basename = os.path.basename(dfg.iloc[1, 11])
  xloc = mpl.dates.MinuteLocator(byminute=range(0,60,1))
  plt.gca().xaxis.set_major_locator(xloc)
  
  # 時刻のフォーマットを設定
  xfmt = mpl.dates.DateFormatter('%m-%d %H:%M:%S')
  plt.gca().xaxis.set_major_formatter(xfmt)
  print(SP)
  ret=plt.show()
# 速度比較グラフ関数
def SpeedGraph2(mode,fileno,filenum=1):
  data_L = df_csv.query('mode == ' + str(mode) )
  # 行選択
  SIZE = 2048 # 1ファイル行数
  SIZEMAX = int(data_L.size/data_L.columns.size)
  FILENO = fileno     
  
  ST = SIZE*FILENO
  SP = SIZE*FILENO+SIZE*filenum-1
  dfg = data_L[ST:SP]
  
  # グラフ作成 　ブレードとフィルムの実速度比較　pSpintor::VAX_speed
  plt.figure(figsize=(10,8))
  ret=plt.plot(dfg['sampletime'], dfg['pCut::CTRL_Position_controller::Actual_speed'],color='b', label="pCut::Actual_speed")
  ret=plt.plot(dfg['sampletime'], dfg['pSvolFilm::CTRL_Position_controller::Actual_speed'], color='r', label="pSvolFilm::Actual_speed")
  ret=plt.plot(dfg['sampletime'], dfg['pSpintor::VAX_speed'], color='g', label="pSpintor::VAX_speed")
  #plt.plot(dfg['timestamp'], dfg['pSpintor::VAX_speed'])
  plt.title(os.path.basename(dfg.iloc[1, 11])+" >>> "+os.path.basename(dfg.iloc[-1, 11]), fontsize=18)
  plt.legend(loc="lower left", fontsize=18) # (3)凡例表示
  
  # ロケータで刻み幅を設定　　basename = os.path.basename(dfg.iloc[1, 11])
  #xloc = mpl.dates.MinuteLocator(byminute=range(0,60,1))
  #plt.gca().xaxis.set_major_locator(xloc)
  
  # 時刻のフォーマットを設定
  xfmt = mpl.dates.DateFormatter('%m-%d %H:%M')
  plt.gca().xaxis.set_major_formatter(xfmt)
  print(SP)
  ret=plt.show()
# モード選択
MODE = 1
number_no = 50
SpeedGraph(fileno=number_no,mode=MODE)
# モード選択
MODE = 1
number_no = 0
SpeedGraph(fileno=number_no,mode=MODE) # 最後のファイルを指定
# モード選択
MODE = 1
dfm = df_csv.query('mode == ' + str(MODE) )# 列選択　
number_file = int(dfm.size/dfm.columns.size/2048)
print("file_number:"+str(number_file))

SpeedGraph(fileno=number_file-1,mode=MODE) # 最後のファイルを指定
# モード選択
MODE = 1

SpeedGraph2(fileno=0,mode=MODE,filenum=2)
# モード選択
MODE = 1

dfm = df_csv.query('mode == ' + str(MODE) )# 列選択　
number_file = int(dfm.size/dfm.columns.size/2048)
print("file_number:"+str(number_file))
SpeedGraph2(fileno=0,mode=MODE,filenum=number_file)

#相関行列
import seaborn
import numpy as np

# モード選択
MODE = 4
dfm = df_csv.query('mode == ' + str(MODE) )# 列選択　

#plt.figure(figsize=(10,10))
df1 = dfm.iloc[:,[5,7,12]]
#df1 = dfm.iloc[:,[5,12]]
#df1 = df1[df1['sampleNumber'] < 3]

ret = seaborn.pairplot(df1, hue='sampleNumber')
#ret = seaborn.pairplot(df1, hue='sampleNumber').savefig('seaborn_pairplot_mode001.png')
from pandas import Series

# モード選択
MODE = 1
dfm = df_csv.query('mode == ' + str(MODE) )# 列選択　

fig = plt.figure()
#ax = fig.add_subplot(1,1,1)

values = dfm.iloc[:,[7]]

st = int(values.size / 4)
values = values[2048*3:2048*(3+4)]
# 棒グラフを描く
values.hist(bins=100, alpha=0.3, color='b', density=True)
# カーネル密度推定
values.plot(kind='kde', style='r--')

plt.show()

from pandas import Series

# モード選択
MODE = 1
dfm = df_csv.query('mode == ' + str(MODE) )# 列選択　

fig = plt.figure()
#ax = fig.add_subplot(1,1,1)

values = dfm.iloc[:,[7]]

st = int(values.size / 2)
values = values[st:st*2]
# 棒グラフを描く
values.hist(bins=100, alpha=0.3, color='b', density=True)
# カーネル密度推定
values.plot(kind='kde', style='r--')

plt.show()

# データ mode割合
buff=[]
df_csv = pd.read_csv('../input/to-csv-outcsv/to_csv_out.csv', parse_dates=[0])
for i in range(1,9):
  MODE = "\"mode" + str(i) + "\"" 
  df = df_csv.query('mode == ' + str(i) )
  temp = ['mode' + str(i),df.size/df.columns.size]
  buff.append(temp)

  print(MODE,str(df.size/df.columns.size))

df2 = pd.DataFrame(buff)
import numpy as np
import matplotlib.pyplot as plt
 
# 円グラフを描画
size = 0.5
res = plt.pie(df2[1],labels=df2[0], autopct='%1.1f%%', radius=3, counterclock=False,startangle=90)
res = plt.legend(loc="upper right", fontsize=10) # (3)凡例表示
df_csv
df1 = df_csv

# グラフ作成
colname = 'mode'
#plt.figure(figsize=(18,4))
plt.plot(df1['sampletime'], df1[colname])
#plt.title(colname, fontsize=18)

# ロケータで刻み幅を設定
#xloc = mpl.dates.MinuteLocator(byminute=range(0,60,1))
#plt.gca().xaxis.set_major_locator(xloc)

# 時刻のフォーマットを設定
#xfmt = mpl.dates.DateFormatter("%H:%M")
#plt.gca().xaxis.set_major_formatter(xfmt)

plt.show()
df1 = df_csv

# グラフ作成
colname = 'pSpintor::VAX_speed'
plt.figure(figsize=(18,4))
plt.plot(df1['sampletime'], df1[colname])
plt.title(colname, fontsize=18)

# ロケータで刻み幅を設定
xloc = mpl.dates.MinuteLocator(byminute=range(0,60,1))
plt.gca().xaxis.set_major_locator(xloc)

# 時刻のフォーマットを設定
xfmt = mpl.dates.DateFormatter("%H:%M")
plt.gca().xaxis.set_major_formatter(xfmt)

plt.show()
# モード選択
MODE = 1
dfm = df_csv.query('mode == ' + str(MODE) )

df1 = df_csv

# グラフ作成
col_name = 'pCut::CTRL_Position_controller::Actual_position'
plt.figure(figsize=(18,4))
plt.scatter(df1['sampletime'], df1[col_name],s=10)
plt.title(col_name, fontsize=18)

# 時刻のフォーマットを設定
xfmt = mpl.dates.DateFormatter("%D %H:%M")
plt.gca().xaxis.set_major_formatter(xfmt)

plt.show()
# モード選択
MODE = 1
dfm = df_csv.query('mode == ' + str(MODE) )

#df1 = df_csv[17000:40000]
df1 = df_csv[100:40000]

# グラフ作成
col_name = 'pCut::CTRL_Position_controller::Actual_position'
plt.figure(figsize=(18,4))
plt.scatter(df1['sampletime'], df1[col_name],s=5)
#plt.plot(pd.Series(list(range(40000-1))),df1[col_name])
plt.title(col_name, fontsize=18)

# 時刻のフォーマットを設定 pd.Series(list(range(40000-1))
xfmt = mpl.dates.DateFormatter('%Y-%m-%d %H:%M')
plt.gca().xaxis.set_major_formatter(xfmt)

plt.show()
# モード選択
MODE = 1
dfm = df_csv.query('mode == ' + str(MODE) )

#df1 = df_csv[17000:40000]
df1 = df_csv

# グラフ作成
col_name = 'mode'
plt.figure(figsize=(18,4))
plt.scatter(df1['sampletime'], df1[col_name],s=5)
plt.title(col_name, fontsize=18)

# 時刻のフォーマットを設定
xfmt = mpl.dates.DateFormatter('%Y-%m-%d %H:%M')
plt.gca().xaxis.set_major_formatter(xfmt)

plt.show()
df1.head(1)
!pip install japanize-matplotlib
import japanize_matplotlib 
import seaborn as sns

sns.set(font="IPAexGothic")
# figureの作成
fig = plt.figure(figsize=(18,4))

# subplotの作成
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()

ret=ax1.plot(df1['sampletime'], df1['mode'], color='b', label="mode")
ret=ax2.plot(df1['sampletime'], df1['pSpintor::VAX_speed'], color='r', label="VAX_speed")

# グラフタイトルとラベル
ax1.set_title("MODE & VAX SPEED")
ax1.set_ylabel("MODE")
ax2.set_ylabel("VAX_speed")

# 凡例
ax1.legend()
ax2.legend(loc='upper center')

ret=plt.show()

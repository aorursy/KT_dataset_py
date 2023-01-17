import numpy as np #นำเข้า numpy แล้วตั้งคำเรียกว่า np
import pandas as pd #นำเข้า pandas แล้วตั้งคำเรียกว่า pd
import matplotlib.pyplot as plt #นำเข้า matplotlib.pyplo แล้วตั้งคำเรียกว่า plt
import seaborn as sns #นำเข้า seaborn แล้วตั้งคำเรียกว่า sns
from sklearn.preprocessing import MinMaxScaler #เรียกใช้ MinMaxScaler จาก sklearn.preprocessing

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# นำเข้าข้อมูล EURUSD_15m_BID_sample โดยจะเรียกใช้ว่า df
df = pd.read_csv('../input/EURUSD_15m_BID_sample.csv')
df.head() #แสดงข้อมูลที่นำเข้ามา
df.info() #แสดงข้อมูลของข้อมูลที่นำเข้ามา
df.count() #จำนวนแถวของแต่ละคอลัมน์
#แสดงจำนวนแถวและคอลัมน์
df.shape
df.index.min(), df.index.max()
df.isna().any() #แสดงความผิดพลาดของข้อมูลโดยผลที่ได้คือ False หมายถึงข้อมูลไม่มีการสูญหาย
#จำนวนของชุดข้อมูลของแต่ละคอลัมน์
df.nunique()
#สร้างข้อมูลheatmap ของข้อมูลที่นำเข้ามาหรือ df 
import matplotlib.pyplot as plt #import matplotlib.pyplot โดยใช้คำสั่งในการเรียกว่า plt

import seaborn as sns #import seaborn โดยใช้คำสั่งในการเรียกว่า sns
corr =df.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
# เปลี่ยนชื่อเพื่อไห้พิมพ์ง่ายขึ้นด้วยคำสั่ง rename โดยเปลี่ยนตัวใหญ่ข้างหน้า
df.rename(columns={'Time' : 'timestamp', 'Open' : 'open', 'Close' : 'close', 
                   'High' : 'high', 'Low' : 'low', 'Close' : 'close', 'Volume' : 'volume'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
df.set_index('timestamp', inplace=True)
df = df.astype(float)
df.head() #เช็คว่าชื่อเปลี่ยนรึยัง
df['low'].head #ดูค่าการขายขั้นต่ำสุดเพื่อนำมาหาค่าเฉลี่ย
df['high'].head #ดูค่าการขายขั้นสูงสุดเพื่อนำมาหาค่า average
#คำนวนหาค่าเฉลี่ย
df['avg_price'] = (df['low'] + df['high'])/2 #นำราคาสูงและต่ำมารวมกันและหารสองจะได้ค่าเฉลี่ยของข้อมูลโดยจะตั้งชื่อของคำสั่งที่ใช้เรียกว่า avg_price
#แสดงค่าเฉลี่ย
df['avg_price'].head 
df['low'].head #แสดงราคาต่ำสุด
df['high'].head #แสดงราคาสูงสุด
df['open'].head #แสดงราคาเปิด
df['close'].head #แสดงราคาปิด
#นำข้อมูลที่ได้มาคำนวณหาค่า OHLC โดยจะนำผลลัพธ์มาแล้วตั้งชื่อเรียกว่า ohlc_price
df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close'])/4

df['ohlc_price'].head #แสดงค่า OHLC
#คำนวณหาค่า diff จากการเปิดและปิดของราคาแล้วตั้งชื่อเรียกว่า oc_diff
df['oc_diff']    = df['open'] - df['close']
df['oc_diff'].head
# หาค่า Range หรือผลต่างของค่าสูงสุดและต่ำสุดของข้อมูล โดยใช้คำเรียกว่า range
df['range']     = df['high'] - df['low']
df['range'].head #แสดงผลต่าง
#สรุปการสร้างข้อมูลใหม่ทั้งหมด

df['momentum']  = df['volume'] * (df['open'] - df['close'])
df['avg_price'] = (df['low'] + df['high'])/2
df['range']     = df['high'] - df['low']
df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close'])/4
df['oc_diff']    = df['open'] - df['close']


#รวมถึงไส่ข้อมูลของ ชั่วโมง วัน และ สัปดาห์ลงไปด้วย

df['hour'] = df.index.hour
df['day']  = df.index.weekday
df['week'] = df.index.week


# เพิ่ม PCA แทนการลดมิติข้อมูลและเป็นการปรับปรุงความแม่นยำเล็กน้อย
from sklearn.decomposition import PCA

dataset = df.copy().values.astype('float32')
pca_features = df.columns.tolist()

pca = PCA(n_components=1)
df['pca'] = pca.fit_transform(dataset)
import matplotlib.colors as colors #เรียกใช้ matplotlib.colors โดยคำเรียกใช้คือ colors
import matplotlib.cm as cm #เรียกใช้ matplotlib.cm โดยคำเรียกใช้คือ cm
import pylab #เรียกใช้ libraly pylab

plt.figure(figsize=(10,5)) #กำหนดขนาด
norm = colors.Normalize(df['ohlc_price'].values.min(), df['ohlc_price'].values.max())
color = cm.viridis(norm(df['ohlc_price'].values))
plt.scatter(df['ohlc_price'].values, df['pca'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)
plt.title('ohlc_price vs pca') #ตั้งชื่อหัวเรื่อง
plt.show()

plt.figure(figsize=(10,5)) #กำหนดขนาด
norm = colors.Normalize(df['volume'].values.min(), df['volume'].values.max())
color = cm.viridis(norm(df['volume'].values))
plt.scatter(df['volume'].values, df['pca'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)
plt.title('volume vs pca') #ตั้งชื่อหัวเรื่อง
plt.show()

plt.figure(figsize=(10,5)) #กำหนดขนาด
norm = colors.Normalize(df['ohlc_price'].values.min(), df['ohlc_price'].values.max())
color = cm.viridis(norm(df['ohlc_price'].values))
plt.scatter(df['ohlc_price'].shift().values, df['pca'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)
plt.title('ohlc_price - 15min future vs pca') #ตั้งชื่อหัวเรื่อง
plt.show()

plt.figure(figsize=(10,5)) #กำหนดขนาด
norm = colors.Normalize(df['volume'].values.min(), df['volume'].values.max())
color = cm.viridis(norm(df['volume'].values))
plt.scatter(df['volume'].shift().values, df['pca'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)
plt.title('volume - 15min future vs pca') #ตั้งชื่อหัวเรื่อง
plt.show()
df.head() # เช็คข้อมูลใหม่อีกครั้งเพื่อดูว่าค่าที่เราใส่ไปไหม่ขึ้นรึยัง
def create_dataset(dataset, look_back=20):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
colormap = plt.cm.inferno
plt.figure(figsize=(15,15))
plt.title('Heat map', y=1.05, size=15)
sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

plt.figure(figsize=(15,5))
corr = df.corr()
sns.heatmap(corr[corr.index == 'close'], linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True);
import cufflinks as cf #เรียกใช้ cufflinks โดยใช้คำสั่งเรียกว่า cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
# นำเข้าข้อมูล EURUSD_15m_BID_sample โดยจะเรียกใช้ว่า df1
df1 = pd.read_csv('../input/EURUSD_15m_BID_sample.csv')
df1.head() #แสดงข้อมูลที่นำเข้ามา
#เพิ่ม MA หรือการขยับของค่าเฉลี่ยเข้าไปในชุดข้อมูลไหม่ที่ถูกตั้งมา
df1['MA'] = df1['Close'].rolling(window=50).mean()  #คำนวณหาค่าการขยับของค่าเฉลี่ยของข้อมูลแล้วตั้งชื่อว่า MA
df1.head(60) #แสดงค่าข้อมูลใหม่ที่เพิ่มเข้ามา

df1.iplot(x='Time',y=['Close','MA']) #สร้างกราฟแสดงการเปรียบเทียบระหว่างราคาจบและการขยับของค่าเฉลี่ย

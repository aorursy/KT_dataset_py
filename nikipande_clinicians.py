import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import seaborn as sb
df4 = pd.read_csv('../input/d-clinic/doctor_corona.csv')
df4.columns = ['state','no.of_staffe']
df4 = df4[:-2]
df4.plot(x='state',y='no.of_staffe',figsize=(20,4),kind='bar',title='No. Of Doctors')
city_wise = pd.read_csv('../input/d-clinic/city_wise_corona.csv')
city_wise.columns=['City','No.OfStafee']
city_wise = city_wise[:-1]
city_wise
city_wise.plot(x='City',y='No.OfStafee',kind='line')
py.show()
from PIL import Image

img = Image.open("../input/ibm-img/Task_9_1.png")
img
img = Image.open("../input/ibm-img/Task_9_2.png")
img
img = Image.open("../input/ibm-img/Task_9_3.png")
img
img = Image.open("../input/ibm-img/Task_9_4.png")
img
img = Image.open("../input/ibm-img/Task_9_5.png")
img
img = Image.open("../input/ibm-img/Task_9_6.png")
img

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

aug15 = pd.read_csv("../input/15-min-data/aug15.csv")

july15 = pd.read_csv("../input/15-min-data/july15.csv")

june15 = pd.read_csv("../input/15-min-data/june15.csv")

oct15 = pd.read_csv("../input/15-min-data/oct15.csv")

sep15 = pd.read_csv("../input/15-min-data/sep15.csv")
frames=[june15,july15,aug15,sep15,oct15]

df=pd.concat(frames, axis=0, join='outer', ignore_index=False)
df['timestamp'] = pd.to_datetime(df['timestamp']) 

df['day']=df['timestamp'].dt.day

df['month'] = df['timestamp'].dt.month

df['dayofweek'] = df['timestamp'].dt.dayofweek #0 monday,1 tuesday,6sunday

df['weekofyear'] = df['timestamp'].dt.weekofyear
#mn

mon= df[df.dayofweek ==0]
pd.crosstab( mon.month,mon.day, rownames=['month'], colnames=['day'])
mon1=mon[(mon.day ==17) & (mon.month==6)]

mon2=mon[(mon.day ==24) & (mon.month==6)]



mon3=mon[(mon.day ==1) & (mon.month==7)]

mon4=mon[(mon.day ==8) & (mon.month==7)]

mon5=mon[(mon.day ==15) & (mon.month==7)]

mon6=mon[(mon.day ==22) & (mon.month==7)]

mon7=mon[(mon.day ==29) & (mon.month==7)]



mon8=mon[(mon.day ==5) & (mon.month==8)]

mon9=mon[(mon.day ==12) & (mon.month==8)]

mon10=mon[(mon.day ==19) & (mon.month==8)]

mon11=mon[(mon.day ==26) & (mon.month==8)]



mon12=mon[(mon.day ==2) & (mon.month==9)]

mon13=mon[(mon.day ==9) & (mon.month==9)]

mon14=mon[(mon.day ==16) & (mon.month==9)]

mon15=mon[(mon.day ==23) & (mon.month==9)]

mon16=mon[(mon.day ==30) & (mon.month==9)]

mon17=mon[(mon.day ==7) & (mon.month==10)]
import plotly.graph_objects as go

fig = go.Figure()

#june

fig.add_trace(go.Scatter(

                y=mon1.mb,

                name="June1",

                line_color='gold',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon2.mb,

                name="June2",

                line_color='yellow',

                opacity=0.8))



#july

fig.add_trace(go.Scatter(

                y=mon3.mb,

                name="July1",

                line_color='lime',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon4.mb,

                name="July2",

                line_color='limegreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon5.mb,

                name="July3",

                line_color='forestgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon6.mb,

                name="July4",

                line_color='seagreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon7.mb,

                name="July5",

                line_color='yellowgreen',

                opacity=0.8))

#aug

fig.add_trace(go.Scatter(

                y=mon8.mb,

                name="Aug1",

                line_color='salmon',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon9.mb,

                name="Aug2",

                line_color='pink',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon10.mb,

                name="Aug3",

                line_color='coral',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon11.mb,

                name="Aug4",

                line_color='mistyrose',

                opacity=0.8))



#sep

fig.add_trace(go.Scatter(

                y=mon12.mb,

                name="Sep1",

                line_color='darkred',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon13.mb,

                name="Sep2",

                line_color='brown',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon14.mb,

                name="Sep3",

                line_color='red',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon15.mb,

                name="Sep4",

                line_color='firebrick',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=mon16.mb,

                name="Sep5",

                line_color='fuchsia',

                opacity=0.8))



#oct

fig.add_trace(go.Scatter(

                y=mon17.mb,

                name="Oct1",

                line_color='blueviolet',

                opacity=0.8))





# Use date string to set xaxis range

#fig.update_layout(xaxis_range=['','2016-12-31'],

                  #title_text="Manually Set Date Range")

fig.show()
tue= df[df.dayofweek ==1]

tue
pd.crosstab( tue.month,tue.day, rownames=['month'], colnames=['day'])
tue1=tue[(tue.day ==18) & (tue.month==6)]

tue2=tue[(tue.day ==25) & (tue.month==6)]



tue3=tue[(tue.day ==2) & (tue.month==7)]

tue4=tue[(tue.day ==9) & (tue.month==7)]

tue5=tue[(tue.day ==16) & (tue.month==7)]

tue6=tue[(tue.day ==23) & (tue.month==7)]

tue7=tue[(tue.day ==30) & (tue.month==7)]



tue8=tue[(tue.day ==6) & (tue.month==8)]

tue9=tue[(tue.day ==13) & (tue.month==8)]

tue10=tue[(tue.day ==20) & (tue.month==8)]

tue11=tue[(tue.day ==27) & (tue.month==8)]



tue12=tue[(tue.day ==3) & (tue.month==9)]

tue13=tue[(tue.day ==10) & (tue.month==9)]

tue14=tue[(tue.day ==17) & (tue.month==9)]

tue15=tue[(tue.day ==24) & (tue.month==9)]



tue16=tue[(tue.day ==1) & (tue.month==10)]

tue17=tue[(tue.day ==8) & (tue.month==10)]
import plotly.graph_objects as go

fig = go.Figure()

#june

fig.add_trace(go.Scatter(

                y=tue1.mb,

                name="June1",

                line_color='gold',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=tue2.mb,

                name="June2",

                line_color='yellow',

                opacity=0.8))



#july

fig.add_trace(go.Scatter(

                y=tue3.mb,

                name="July1",

                line_color='lime',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=tue4.mb,

                name="July2",

                line_color='limegreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=tue5.mb,

                name="July3",

                line_color='forestgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=tue6.mb,

                name="July4",

                line_color='seagreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=tue7.mb,

                name="July5",

                line_color='yellowgreen',

                opacity=0.8))

#aug

fig.add_trace(go.Scatter(

                y=tue8.mb,

                name="Aug1",

                line_color='salmon',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=tue9.mb,

                name="Aug2",

                line_color='pink',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=tue10.mb,

                name="Aug3",

                line_color='coral',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=tue11.mb,

                name="Aug4",

                line_color='mistyrose',

                opacity=0.8))



#sep

fig.add_trace(go.Scatter(

                y=tue12.mb,

                name="Sep1",

                line_color='darkred',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=tue13.mb,

                name="Sep2",

                line_color='brown',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=tue14.mb,

                name="Sep3",

                line_color='red',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=tue15.mb,

                name="Sep4",

                line_color='firebrick',

                opacity=0.8))





#oct

fig.add_trace(go.Scatter(

                y=tue16.mb,

                name="Oct1",

                line_color='fuchsia',

                opacity=0.8))



fig.add_trace(go.Scatter(

                y=tue17.mb,

                name="Oct2",

                line_color='blueviolet',

                opacity=0.8))





# Use date string to set xaxis range

#fig.update_layout(xaxis_range=['','2016-12-31'],

                  #title_text="Manually Set Date Range")

fig.show()
wed= df[df.dayofweek ==2]

pd.crosstab( wed.month,wed.day, rownames=['month'], colnames=['day'])
wed1=wed[(wed.day ==12) & (wed.month==6)]

wed2=wed[(wed.day ==19) & (wed.month==6)]

wed3=wed[(wed.day ==26) & (wed.month==6)]



wed4=wed[(wed.day ==3) & (wed.month==7)]

wed5=wed[(wed.day ==10) & (wed.month==7)]

wed6=wed[(wed.day ==17) & (wed.month==7)]

wed7=wed[(wed.day ==24) & (wed.month==7)]

wed8=wed[(wed.day ==31) & (wed.month==7)]



wed9=wed[(wed.day ==7) & (wed.month==8)]

wed10=wed[(wed.day ==14) & (wed.month==8)]

wed11=wed[(wed.day ==21) & (wed.month==8)]

wed12=wed[(wed.day ==28) & (wed.month==8)]



wed13=wed[(wed.day ==4) & (wed.month==9)]

wed14=wed[(wed.day ==11) & (wed.month==9)]

wed15=wed[(wed.day ==18) & (wed.month==9)]

wed16=wed[(wed.day ==25) & (wed.month==9)]



wed17=wed[(wed.day ==2) & (wed.month==10)]

wed18=wed[(wed.day ==9) & (wed.month==10)]





import plotly.graph_objects as go

fig = go.Figure()

#june

fig.add_trace(go.Scatter(

                y=wed1.mb,x=[40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,

                            70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96],

                name="June1",

                line_color='dimgrey',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed2.mb,

                name="June2",

                line_color='gold',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed3.mb,

                name="June3",

                line_color='yellow',

                opacity=0.8))



#july

fig.add_trace(go.Scatter(

                y=wed4.mb,

                name="July1",

                line_color='lime',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed5.mb,

                name="July2",

                line_color='limegreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed6.mb,

                name="July3",

                line_color='forestgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed7.mb,

                name="July4",

                line_color='seagreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed8.mb,

                name="July5",

                line_color='yellowgreen',

                opacity=0.8))



#aug

fig.add_trace(go.Scatter(

                y=wed9.mb,

                name="Aug1",

                line_color='salmon',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed10.mb,

                name="Aug2",

                line_color='pink',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed11.mb,

                name="Aug3",

                line_color='coral',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed12.mb,

                name="Aug4",

                line_color='mistyrose',

                opacity=0.8))



#sep

fig.add_trace(go.Scatter(

                y=wed13.mb,

                name="Sep1",

                line_color='darkred',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed14.mb,

                name="Sep2",

                line_color='brown',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed15.mb,

                name="Sep3",

                line_color='red',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=wed16.mb,

                name="Sep4",

                line_color='firebrick',

                opacity=0.8))





#oct

fig.add_trace(go.Scatter(

                y=wed17.mb,

                name="Oct1",

                line_color='fuchsia',

                opacity=0.8))



fig.add_trace(go.Scatter(

                y=wed18.mb,

                name="Oct2",

                line_color='blueviolet',

                opacity=0.8))





# Use date string to set xaxis range

#fig.update_layout(xaxis_range=['','2016-12-31'],

                  #title_text="Manually Set Date Range")

fig.show()
t= df[df.dayofweek ==3]

pd.crosstab( t.month,t.day, rownames=['month'], colnames=['day'])
t1=t[(t.day ==13) & (t.month==6)]

t2=t[(t.day ==20) & (t.month==6)]

t3=t[(t.day ==27) & (t.month==6)]



t4=t[(t.day ==4) & (t.month==7)]

t5=t[(t.day ==11) & (t.month==7)]

t6=t[(t.day ==18) & (t.month==7)]

t7=t[(t.day ==25) & (t.month==7)]



t8=t[(t.day ==1) & (t.month==8)]

t9=t[(t.day ==8) & (t.month==8)]

t10=t[(t.day ==15) & (t.month==8)]

t11=t[(t.day ==22) & (t.month==8)]

t12=t[(t.day ==29) & (t.month==8)]



t13=t[(t.day ==5) & (t.month==9)]

t14=t[(t.day ==12) & (t.month==9)]

t15=t[(t.day ==19) & (t.month==9)]

t16=t[(t.day ==26) & (t.month==9)]



t17=t[(t.day ==3) & (t.month==10)]

t18=t[(t.day ==10) & (t.month==10)]





import plotly.graph_objects as go

fig = go.Figure()

#june

fig.add_trace(go.Scatter(

                y=t1.mb,

                name="June1",

                line_color='dimgrey',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t2.mb,

                name="June2",

                line_color='gold',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t3.mb,

                name="June3",

                line_color='yellow',

                opacity=0.8))



#july

fig.add_trace(go.Scatter(

                y=t4.mb,

                name="July1",

                line_color='lime',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t5.mb,

                name="July2",

                line_color='limegreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t6.mb,

                name="July3",

                line_color='forestgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t7.mb,

                name="July4",

                line_color='seagreen',

                opacity=0.8))





#aug

fig.add_trace(go.Scatter(

                y=t8.mb,

                name="Aug1",

                line_color='yellowgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t9.mb,

                name="Aug2",

                line_color='salmon',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t10.mb,

                name="Aug3",

                line_color='pink',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t11.mb,

                name="Aug4",

                line_color='coral',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t12.mb,

                name="Aug5",

                line_color='mistyrose',

                opacity=0.8))



#sep

fig.add_trace(go.Scatter(

                y=t13.mb,

                name="Sep1",

                line_color='darkred',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t14.mb,

                name="Sep2",

                line_color='brown',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t15.mb,

                name="Sep3",

                line_color='red',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t16.mb,

                name="Sep4",

                line_color='firebrick',

                opacity=0.8))





#oct

fig.add_trace(go.Scatter(

                y=t17.mb,

                name="Oct1",

                line_color='fuchsia',

                opacity=0.8))



fig.add_trace(go.Scatter(

                y=t18.mb,

                name="Oct2",

                line_color='blueviolet',

                opacity=0.8))





# Use date string to set xaxis range

#fig.update_layout(xaxis_range=['','2016-12-31'],

                  #title_text="Manually Set Date Range")

fig.show()
t= df[df.dayofweek ==4]

pd.crosstab( t.month,t.day, rownames=['month'], colnames=['day'])
t1=t[(t.day ==14) & (t.month==6)]

t2=t[(t.day ==21) & (t.month==6)]

t3=t[(t.day ==28) & (t.month==6)]



t4=t[(t.day ==5) & (t.month==7)]

t5=t[(t.day ==12) & (t.month==7)]

t6=t[(t.day ==19) & (t.month==7)]

t7=t[(t.day ==26) & (t.month==7)]



t8=t[(t.day ==2) & (t.month==8)]

t9=t[(t.day ==9) & (t.month==8)]

t10=t[(t.day ==16) & (t.month==8)]

t11=t[(t.day ==23) & (t.month==8)]

t12=t[(t.day ==30) & (t.month==8)]



t13=t[(t.day ==6) & (t.month==9)]

t14=t[(t.day ==13) & (t.month==9)]

t15=t[(t.day ==20) & (t.month==9)]

t16=t[(t.day ==27) & (t.month==9)]



t17=t[(t.day ==4) & (t.month==10)]

t18=t[(t.day ==11) & (t.month==10)]
import plotly.graph_objects as go

fig = go.Figure()

#june

fig.add_trace(go.Scatter(

                y=t1.mb,

                name="June1",

                line_color='dimgrey',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t2.mb,

                name="June2",

                line_color='gold',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t3.mb,

                name="June3",

                line_color='yellow',

                opacity=0.8))



#july

fig.add_trace(go.Scatter(

                y=t4.mb,

                name="July1",

                line_color='lime',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t5.mb,

                name="July2",

                line_color='limegreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t6.mb,

                name="July3",

                line_color='forestgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t7.mb,

                name="July4",

                line_color='seagreen',

                opacity=0.8))





#aug

fig.add_trace(go.Scatter(

                y=t8.mb,

                name="Aug1",

                line_color='yellowgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t9.mb,

                name="Aug2",

                line_color='salmon',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t10.mb,

                name="Aug3",

                line_color='pink',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t11.mb,

                name="Aug4",

                line_color='coral',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t12.mb,

                name="Aug5",

                line_color='mistyrose',

                opacity=0.8))



#sep

fig.add_trace(go.Scatter(

                y=t13.mb,

                name="Sep1",

                line_color='darkred',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t14.mb,

                name="Sep2",

                line_color='brown',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t15.mb,

                name="Sep3",

                line_color='red',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t16.mb,

                name="Sep4",

                line_color='firebrick',

                opacity=0.8))





#oct

fig.add_trace(go.Scatter(

                y=t17.mb,

                name="Oct1",

                line_color='fuchsia',

                opacity=0.8))



fig.add_trace(go.Scatter(

                y=t18.mb,

                name="Oct2",

                line_color='blueviolet',

                opacity=0.8))





# Use date string to set xaxis range

#fig.update_layout(xaxis_range=['','2016-12-31'],

                  #title_text="Manually Set Date Range")

fig.show()
t= df[df.dayofweek ==5]

pd.crosstab( t.month,t.day, rownames=['month'], colnames=['day'])
t1=t[(t.day ==15) & (t.month==6)]

t2=t[(t.day ==22) & (t.month==6)]

t3=t[(t.day ==29) & (t.month==6)]



t4=t[(t.day ==6) & (t.month==7)]

t5=t[(t.day ==13) & (t.month==7)]

t6=t[(t.day ==20) & (t.month==7)]

t7=t[(t.day ==27) & (t.month==7)]



t8=t[(t.day ==3) & (t.month==8)]

t9=t[(t.day ==10) & (t.month==8)]

t10=t[(t.day ==17) & (t.month==8)]

t11=t[(t.day ==24) & (t.month==8)]

t12=t[(t.day ==31) & (t.month==8)]



t13=t[(t.day ==7) & (t.month==9)]

t14=t[(t.day ==14) & (t.month==9)]

t15=t[(t.day ==21) & (t.month==9)]

t16=t[(t.day ==28) & (t.month==9)]



t17=t[(t.day ==5) & (t.month==10)]

import plotly.graph_objects as go

fig = go.Figure()

#june

fig.add_trace(go.Scatter(

                y=t1.mb,

                name="June1",

                line_color='dimgrey',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t2.mb,

                name="June2",

                line_color='gold',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t3.mb,

                name="June3",

                line_color='yellow',

                opacity=0.8))



#july

fig.add_trace(go.Scatter(

                y=t4.mb,

                name="July1",

                line_color='lime',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t5.mb,

                name="July2",

                line_color='limegreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t6.mb,

                name="July3",

                line_color='forestgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t7.mb,

                name="July4",

                line_color='seagreen',

                opacity=0.8))





#aug

fig.add_trace(go.Scatter(

                y=t8.mb,

                name="Aug1",

                line_color='yellowgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t9.mb,

                name="Aug2",

                line_color='salmon',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t10.mb,

                name="Aug3",

                line_color='pink',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t11.mb,

                name="Aug4",

                line_color='coral',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t12.mb,

                name="Aug5",

                line_color='mistyrose',

                opacity=0.8))



#sep

fig.add_trace(go.Scatter(

                y=t13.mb,

                name="Sep1",

                line_color='darkred',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t14.mb,

                name="Sep2",

                line_color='brown',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t15.mb,

                name="Sep3",

                line_color='red',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t16.mb,

                name="Sep4",

                line_color='firebrick',

                opacity=0.8))





#oct

fig.add_trace(go.Scatter(

                y=t17.mb,

                name="Oct1",

                line_color='fuchsia',

                opacity=0.8))









# Use date string to set xaxis range

#fig.update_layout(xaxis_range=['','2016-12-31'],

                  #title_text="Manually Set Date Range")

fig.show()
t= df[df.dayofweek ==6]

pd.crosstab( t.month,t.day, rownames=['month'], colnames=['day'])
t1=t[(t.day ==16) & (t.month==6)]

t2=t[(t.day ==23) & (t.month==6)]

t3=t[(t.day ==30) & (t.month==6)]



t4=t[(t.day ==7) & (t.month==7)]

t5=t[(t.day ==14) & (t.month==7)]

t6=t[(t.day ==21) & (t.month==7)]

t7=t[(t.day ==28) & (t.month==7)]



t8=t[(t.day ==4) & (t.month==8)]

t9=t[(t.day ==11) & (t.month==8)]

t10=t[(t.day ==18) & (t.month==8)]

t11=t[(t.day ==25) & (t.month==8)]



t12=t[(t.day ==1) & (t.month==9)]

t13=t[(t.day ==8) & (t.month==9)]

t14=t[(t.day ==15) & (t.month==9)]

t15=t[(t.day ==22) & (t.month==9)]

t16=t[(t.day ==29) & (t.month==9)]





t17=t[(t.day ==5) & (t.month==10)]

import plotly.graph_objects as go

fig = go.Figure()

#june

fig.add_trace(go.Scatter(

                y=t1.mb,

                name="June1",

                line_color='dimgrey',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t2.mb,

                name="June2",

                line_color='gold',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t3.mb,

                name="June3",

                line_color='yellow',

                opacity=0.8))



#july

fig.add_trace(go.Scatter(

                y=t4.mb,

                name="July1",

                line_color='lime',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t5.mb,

                name="July2",

                line_color='limegreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t6.mb,

                name="July3",

                line_color='forestgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t7.mb,

                name="July4",

                line_color='seagreen',

                opacity=0.8))





#aug

fig.add_trace(go.Scatter(

                y=t8.mb,

                name="Aug1",

                line_color='yellowgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t9.mb,

                name="Aug2",

                line_color='salmon',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t10.mb,

                name="Aug3",

                line_color='pink',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t11.mb,

                name="Aug4",

                line_color='coral',

                opacity=0.8))





#sep

fig.add_trace(go.Scatter(

                y=t12.mb,

                name="Sep1",

                line_color='mistyrose',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t13.mb,

                name="Sep2",

                line_color='darkred',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t14.mb,

                name="Sep3",

                line_color='brown',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t15.mb,

                name="Sep4",

                line_color='red',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=t16.mb,

                name="Sep5",

                line_color='firebrick',

                opacity=0.8))





#oct

fig.add_trace(go.Scatter(

                y=t17.mb,

                name="Oct1",

                line_color='fuchsia',

                opacity=0.8))









# Use date string to set xaxis range

#fig.update_layout(xaxis_range=['','2016-12-31'],

                  #title_text="Manually Set Date Range")

fig.show()


#df1=df.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()



# df1=df.resample('D', on = 'timestamp').sum()

# # df1



# df['timestamp'] = pd.to_datetime(df['timestamp'])  # Makes sure your timestamp is in datetime format

# df=df.groupby(pd.Grouper(key='timestamp', freq='1D')).sum()

# df







#df1.set_index('weekofyear')
#df.to_csv('dailysummrized_forallmonths')
import pandas as pd

df_daily = pd.read_csv("../input/daily-summarized/dailysummrized_forallmonths")

# df_daily.set_index('timestamp')
df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp']) 

df_daily['month'] = df_daily['timestamp'].dt.month

df_daily['weekofyear'] = df_daily['timestamp'].dt.weekofyear
pd.crosstab( df_daily.month,df_daily.weekofyear, rownames=['month'], colnames=['weekofyear'])
week1=df_daily[df_daily.weekofyear ==24]

week2=df_daily[df_daily.weekofyear ==25]

week3=df_daily[df_daily.weekofyear ==26]

week4=df_daily[df_daily.weekofyear ==27]

week5=df_daily[df_daily.weekofyear ==28]

week6=df_daily[df_daily.weekofyear ==29]



week7=df_daily[df_daily.weekofyear ==30]

week8=df_daily[df_daily.weekofyear ==31]

week9=df_daily[df_daily.weekofyear ==32]

week10=df_daily[df_daily.weekofyear ==33]

week11=df_daily[df_daily.weekofyear ==34]

week12=df_daily[df_daily.weekofyear ==35]



week13=df_daily[df_daily.weekofyear ==36]

week14=df_daily[df_daily.weekofyear ==37]

week15=df_daily[df_daily.weekofyear ==38]

week16=df_daily[df_daily.weekofyear ==39]

week17=df_daily[df_daily.weekofyear ==40]

week18=df_daily[df_daily.weekofyear ==41]
import plotly.graph_objects as go

fig = go.Figure()

#juen

fig.add_trace(go.Scatter(

                y=week1.mb,x=[2,3,4,5,6],

                name="June_week1",

                line_color='dimgrey',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week2.mb,

                name="June_week2",

                line_color='gold',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week3.mb,

                name="June_week3",

                line_color='yellow',

                opacity=0.8))



#july

fig.add_trace(go.Scatter(

                y=week4.mb,

                name="July_week1",

                line_color='lime',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week5.mb,

                name="July_week2",

                line_color='limegreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week6.mb,

                name="July_week3",

                line_color='forestgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week7.mb,

                name="July_week4",

                line_color='yellowgreen',

                opacity=0.8))





#aug

fig.add_trace(go.Scatter(

                y=week8.mb,

                name="Aug_week1",

                line_color='pink',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week9.mb,

                name="Aug_week2",

                line_color='salmon',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week10.mb,

                name="Aug_week3",

                line_color='coral',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week11.mb,

                name="Aug_week4",

                line_color='mistyrose',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week12.mb,

                name="Aug_week5",

                line_color='peru',

                opacity=0.8))





#sep

fig.add_trace(go.Scatter(

                y=week13.mb,

                name="Sep_week1",

                line_color='darkred',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week14.mb,

                name="Sep_week2",

                line_color='brown',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week15.mb,

                name="Sep_week3",

                line_color='red',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week16.mb,

                name="Sep_week4",

                line_color='firebrick',

                opacity=0.8))

#oct

fig.add_trace(go.Scatter(

                y=week17.mb,

                name="Oct_week1",

                line_color='blueviolet',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week18.mb,

                name="Oct_week2",

                line_color='orchid',

                opacity=0.8))





fig.show()
# import plotly.graph_objects as go



# fig = go.Figure(go.Scatter(

    

    

#     x = ["Mon", "Tue", "Wed","Thu","Fri","Sat","Sun"]))



# fig.update_layout(

#     xaxis = go.layout.XAxis(

#         tickangle = 90,

#         title_text = "Daily usage",

#         title_font = {"size": 20},

#         title_standoff = 25),

#     )



# fig.show()
# i=range(18)

# for i in range(18):

# k ='week'+i.astype(str)

# k =pd.DataFrame()




# for i in range (18):

#  k='weeek'+i.astype(str)

#  k=df_daily[df_daily.weekofyear ==i+24]

import plotly.graph_objects as go

fig = go.Figure()

#juen

fig.add_trace(go.Scatter(

                y=week1.mb,x=[2,3,4,5,6],

                name="June_week1",

                line_color='dimgrey',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week2.mb,

                name="June_week2",

                line_color='gold',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week3.mb,

                name="June_week3",

                line_color='yellow',

                opacity=0.8))

fig.show()
import plotly.graph_objects as go

fig = go.Figure()

#july

fig.add_trace(go.Scatter(

                y=week4.mb,

                name="July_week1",

                line_color='lime',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week5.mb,

                name="July_week2",

                line_color='limegreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week6.mb,

                name="July_week3",

                line_color='forestgreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week7.mb,

                name="July_week4",

                line_color='yellowgreen',

                opacity=0.8))

fig.show()
import plotly.graph_objects as go

fig = go.Figure()



#aug

fig.add_trace(go.Scatter(

                y=week8.mb,

                name="Aug_week1",

                line_color='pink',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week9.mb,

                name="Aug_week2",

                line_color='salmon',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week10.mb,

                name="Aug_week3",

                line_color='coral',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week11.mb,

                name="Aug_week4",

                line_color='mistyrose',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week12.mb,

                name="Aug_week5",

                line_color='peru',

                opacity=0.8))









fig.show()
import plotly.graph_objects as go

fig = go.Figure()

#sep

fig.add_trace(go.Scatter(

                y=week13.mb,

                name="Sep_week1",

                line_color='darkred',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week14.mb,

                name="Sep_week2",

                line_color='brown',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week15.mb,

                name="Sep_week3",

                line_color='red',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week16.mb,

                name="Sep_week4",

                line_color='firebrick',

                opacity=0.8))







fig.show()
import plotly.graph_objects as go

fig = go.Figure()

#oct

fig.add_trace(go.Scatter(

                y=week17.mb,

                name="Oct_week1",

                line_color='blueviolet',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=week18.mb,

                name="Oct_week2",

                line_color='orchid',

                opacity=0.8))

fig.show()
df_daily
June=df_daily[df_daily.month==6]

July=df_daily[df_daily.month==7]

Aug=df_daily[df_daily.month==8]

Sep=df_daily[df_daily.month==9]

Oct=df_daily[df_daily.month==10]
import plotly.graph_objects as go

fig = go.Figure()

#oct

fig.add_trace(go.Scatter(

                y=June.mb,

                name="June",x=[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],

                line_color='gold',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=July.mb,x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],

                name="July",

                line_color='green',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=Aug.mb,x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],

                name="Aug",

                line_color='coral',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=Sep.mb,x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],

                name="Sep",

                line_color='red',

                opacity=0.8))

fig.add_trace(go.Scatter(

                y=Oct.mb,x=[1,2,3,4,5,6,7,8,9,10,11],

                name="Oct",

                line_color='blueviolet',

                opacity=0.8))

fig.show()


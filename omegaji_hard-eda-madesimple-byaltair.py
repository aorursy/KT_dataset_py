



import numpy as np

import pandas as pd 



df=pd.read_csv("/kaggle/input/udemy-courses/udemy_courses.csv")
import altair as alt

import altair_render_script

sortdf=df[df["is_paid"]==True].sort_values(["num_subscribers"],ascending=False)[:20]

alt.Chart(sortdf).mark_bar().encode(alt.X("course_title"),alt.Y("num_subscribers"),tooltip=["course_title","num_subscribers"]).properties(width=600)

#alt.Chart().mark_bar().encode(x=[1,2,3,4],y=[10,20,30,40])

from datetime import datetime

def extractdate(x):

    return datetime.strptime(x[:10],"%Y-%m-%d")

df["day"]=df["published_timestamp"].apply(extractdate)

df["day"]=df["day"].apply(lambda x: int(x.day))



df["month"]=df["published_timestamp"].apply(extractdate)

df["month"]=df["month"].apply(lambda x: int(x.month))



df["day_in_year"]=df["published_timestamp"].apply(extractdate)

df["day_in_year"]=df["day_in_year"].apply(lambda x: int(x.timetuple().tm_yday))





df["year"]=df["published_timestamp"].apply(extractdate)

df["year"]=df["year"].apply(lambda x: int(x.year))


courses=df["subject"].unique()



slider2 = alt.binding_range(min=2011, max=2017, step=1)

select_year= alt.selection_single(name='year', fields=['year'],

                                   bind=slider2, init={'year': 2016})

base = alt.Chart(df[df["subject"]==courses[0]].groupby(["month","year"]).count().reset_index()).add_selection(select_year).transform_filter(select_year)



c1=base.mark_line(point=True).encode(alt.X("month"), alt.Y("course_title",title="Buisness Finance Courses made"),tooltip=alt.Tooltip(["course_title"],title="Buisness Finance Courses made"))

base = alt.Chart(df[df["subject"]==courses[1]].groupby(["month","year"]).count().reset_index()).add_selection(select_year).transform_filter(select_year)



c2=base.mark_line(point=True).encode(alt.X("month"), alt.Y("course_title",title="Graphic Design Courses made"),tooltip=alt.Tooltip(["course_title"],title="Graphic Design Courses made"))







base = alt.Chart(df[df["subject"]==courses[2]].groupby(["month","year"]).count().reset_index()).add_selection(select_year).transform_filter(select_year)



c3=base.mark_line(point=True).encode(alt.X("month"), alt.Y("course_title",title="Musical Instruments Courses made"),tooltip=alt.Tooltip(["course_title"],title="Musical Instruments Courses made"))







base = alt.Chart(df[df["subject"]==courses[3]].groupby(["month","year"]).count().reset_index()).add_selection(select_year).transform_filter(select_year)



c4=base.mark_line(point=True).encode(alt.X("month"), alt.Y("course_title",title="Web Development Courses made"),tooltip=alt.Tooltip(["course_title"],title="Web Development Courses made"))

alt.vconcat(alt.concat(c1,c2,spacing=80),alt.concat(c3,c4,spacing=80),spacing=5)



courses=df["subject"].unique()

slider2 = alt.binding_range(min=2011, max=2017, step=1)

select_year= alt.selection_single(name='year', fields=['year'],

                                   bind=slider2, init={'year': 2016})

base = alt.Chart(df[df["subject"]==courses[0]]).add_selection(select_year).transform_filter(select_year)







a=base.mark_bar(size=10).encode(

    alt.X("month"),alt.Y("sum(price)"),color="level",tooltip=alt.Tooltip(["sum(price)"],title="Total Price")).properties(title="Buisness Finance Prices over the month")



b=base.mark_bar(size=10).encode(

    alt.X("month"),alt.Y("sum(price)"),color="level",tooltip=alt.Tooltip(["sum(price)"],title="Total Price")).properties(title="Graph Design Prices over the month")



c=base.mark_bar(size=10).encode(

    alt.X("month"),alt.Y("sum(price)"),color="level",tooltip=alt.Tooltip(["sum(price)"],title="Total Price")).properties(title="Musical Instrument Prices over the month")



d=base.mark_bar(size=10).encode(

    alt.X("month"),alt.Y("sum(price)"),color="level",tooltip=alt.Tooltip(["sum(price)"],title="Total Price")).properties(title="Web Development Prices over the month")

alt.vconcat(alt.concat(a,b,spacing=20),alt.concat(c,d,spacing=20),spacing=5)

sortdf=df[df["is_paid"]==False].sort_values(["num_subscribers"],ascending=False)[:20]

alt.Chart(sortdf).mark_bar().encode(alt.X("course_title"),alt.Y("num_subscribers"),tooltip=["course_title","num_subscribers"]).properties(width=600)



alt.Chart(df).mark_bar().encode(

    column='level',

    x='num_subscribers',

    y='subject' ,tooltip=["num_subscribers"]

).properties(width=220)

alt.Chart(df).mark_bar().encode(

   column="level",

    x='num_lectures',

    y='subject' ,tooltip=["num_lectures"]

).properties(width=220)

slider = alt.binding_range(min=1, max=12, step=1)

select_month = alt.selection_single(name='month', fields=['month'],

                                   bind=slider, init={'month': 1})



slider2 = alt.binding_range(min=2011, max=2017, step=1)

select_year= alt.selection_single(name='year', fields=['year'],

                                   bind=slider2, init={'year': 2016})

base = alt.Chart(df).add_selection(select_year,select_month).transform_filter(select_year).transform_filter(

    select_month

)









left = base.transform_filter(alt.datum.is_paid==True).encode( 

     y=alt.Y('subject'),

     x=alt.X('sum(num_subscribers)',

            

            title='NumOfSubscribers')

    ,tooltip=["sum(num_subscribers)","subject"]).mark_bar(size=20).properties(title='subscribers Over the month for PAID',height=200)



left



right = base.transform_filter(alt.datum.is_paid==False).encode(

     y=alt.Y('subject'),

    x=alt.X('sum(num_subscribers)'),tooltip=["sum(num_subscribers)","subject"]).mark_bar(size=20).properties(title='subscribers Over the month for FREE',height=200)





right




alt.Chart(df).mark_bar(size=10).encode(

    alt.X("num_lectures:Q", bin=alt.Bin(step=21)),

    alt.Y("count(subject)",title="subject count "),

    row='level',color='subject',tooltip=["count(subject)","subject","num_lectures"]

).properties(width=700)




alt.Chart(df).mark_bar(size=13).encode(

    alt.X("content_duration:Q", bin=alt.Bin(step=2)),

    alt.Y("count(subject)",title="subject count "),

    row='level',color='subject',tooltip=["count(subject)","subject","content_duration"]

).properties(width=700)
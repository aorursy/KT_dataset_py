import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
udemy = pd.read_csv('../input/udemy-courses/udemy_courses.csv')
udemy.head(5)
udemy.info()
udemy.isna().sum()
udemy.duplicated().sum()
udemy = udemy.drop_duplicates().reset_index(drop=True)
udemy.duplicated().sum()
udemy[udemy['num_lectures']==0]['course_id'].count()
udemy.loc[udemy['num_lectures']==0]
udemy.drop([890], inplace = True)
udemy.reset_index(drop=True, inplace=True)
udemy[udemy['num_lectures']==0]['course_id'].count()
udemy.info()
udemy['subject'].unique()
udemy_bf = udemy.loc[
    (udemy['subject'] == 'Business Finance') & (udemy['is_paid'] == False)]

udemy_gd = udemy.loc[
    (udemy['subject'] == 'Graphic Design') & (udemy['is_paid'] == False)]

udemy_mi = udemy.loc[
    (udemy['subject'] == 'Musical Instruments') & (udemy['is_paid'] == False)]

udemy_wd = udemy.loc[
    (udemy['subject'] == 'Web Development') & (udemy['is_paid'] == False)]
def find_best_courses(df):
    return df.sort_values('num_subscribers', ascending=False)[
       ['course_id', 'course_title', 'level', 'num_subscribers', 'num_reviews', 'subject']].head()
# <sort the free Business Finance courses in descending order by 'num_subscribers' column and show the first 5 of them>
find_best_courses(udemy_bf)
# <sort the free Graphic Design courses in descending order by 'num_subscribers' column and show the first 5 of them>
find_best_courses(udemy_gd)
# <sort the free Musical Instruments courses in descending order by 'num_subscribers' column and show the first 5 of them>
find_best_courses(udemy_mi)
# <sort the free Web Development courses in descending order by 'num_subscribers' column and show the first 5 of them>
find_best_courses(udemy_wd)
data = [
    [udemy.loc[492][0], udemy.loc[492][1], udemy.loc[492][5], 
     udemy.loc[492][8], udemy.loc[492][9], udemy.loc[492][11]],
    [udemy.loc[2820][0], udemy.loc[2820][1], udemy.loc[2820][5], 
     udemy.loc[2820][8], udemy.loc[2820][9], udemy.loc[2820][11]],
    [udemy.loc[1456][0], udemy.loc[1456][1], udemy.loc[1456][5], 
     udemy.loc[1456][8], udemy.loc[1456][9], udemy.loc[1456][11]],
    [udemy.loc[1890][0], udemy.loc[1890][1], udemy.loc[1890][5], 
      udemy.loc[1890][8], udemy.loc[1890][9], udemy.loc[1890][11]]
]

columns = ['course_id', 'course_title', 'num_subscribers', 'level', 'content_duration', 'subject']
best_courses_per_subject = pd.DataFrame(data = data, columns = columns)
best_courses_per_subject
udemy['public_interest'] = udemy['num_subscribers'] + udemy['num_reviews']
udemy.head()
udemy.sort_values('public_interest', ascending=False).head()
udemy_bf = udemy.loc[(udemy['subject'] == 'Business Finance')]
udemy_gd = udemy.loc[(udemy['subject'] == 'Graphic Design')]
udemy_mi = udemy.loc[(udemy['subject'] == 'Musical Instruments')]
udemy_wd = udemy.loc[(udemy['subject'] == 'Web Development')]
# <Business Finance mean price>
bf_mean_price = udemy_bf['price'].mean()

# <Graphic Design mean price>
gd_mean_price = udemy_gd['price'].mean()

# <Musical Instruments mean price>
mi_mean_price = udemy_mi['price'].mean()

# <Web Development mean price>
wd_mean_price = udemy_wd['price'].mean()

print('Business Finance mean price: {:.3f}'.format(bf_mean_price))
print('Graphic Design mean price: {:.3f}'.format(gd_mean_price))
print('Musical Instruments mean price: {:.3f}'.format(mi_mean_price))
print('Web Development mean price: {:.3f}'.format(wd_mean_price))
# <Business Finance mean content duration>
bf_mean_con_dur = udemy_bf['content_duration'].mean()

# <Graphic Design mean content duration>
gd_mean_con_dur = udemy_gd['content_duration'].mean()

# <Musical Instruments mean content duration>
mi_mean_con_dur = udemy_mi['content_duration'].mean()

# <Web Development mean content duration>
wd_mean_con_dur = udemy_wd['content_duration'].mean()

print('Business Finance mean content duration: {:.3f}'.format(bf_mean_con_dur))
print('Graphic Design mean content duration: {:.3f}'.format(gd_mean_con_dur))
print('Musical Instruments mean content duration: {:.3f}'.format(mi_mean_con_dur))
print('Web Development mean content duration: {:.3f}'.format(wd_mean_con_dur))
# <Business Finance mean index of public interest>
bf_mean_ipi = udemy_bf['public_interest'].mean() 

# <Graphic Design mean index of public interest>
gd_mean_ipi = udemy_gd['public_interest'].mean() 

# <Musical Instruments mean index of public interest>
mi_mean_ipi = udemy_mi['public_interest'].mean()

# <Web Development mean index of public interest>
wd_mean_ipi = udemy_wd['public_interest'].mean() 

print('Business Finance mean public interest index: {:.3f}'.format(bf_mean_ipi))
print('Graphic Design mean public interest index: {:.3f}'.format(gd_mean_ipi))
print('Musical Instruments mean public interest index: {:.3f}'.format(mi_mean_ipi))
print('Web Development mean public interest index: {:.3f}'.format(wd_mean_ipi))
bf_eng = udemy.loc[
    (udemy['subject'] == 'Business Finance') &
    (udemy['price'] <= bf_mean_price) & 
    (udemy['content_duration'] >= bf_mean_con_dur) &
    (udemy['public_interest'] >= bf_mean_ipi)
]

print('Number of courses: ', bf_eng['course_id'].count())
#The most engaging Business Finance courses rows
bf_eng
gd_eng = udemy.loc[
    (udemy['subject'] == 'Graphic Design') &
    (udemy['price'] <= gd_mean_price) & 
    (udemy['content_duration'] >= gd_mean_con_dur) &
    (udemy['public_interest'] >= gd_mean_ipi)
]

print('Number of courses: ', gd_eng['course_id'].count())
#The most engaging Graphic Design courses rows
gd_eng
mi_eng = udemy.loc[
    (udemy['subject'] == 'Musical Instruments') &
    (udemy['price'] <= mi_mean_price) & 
    (udemy['content_duration'] >= mi_mean_con_dur) &
    (udemy['public_interest'] >= mi_mean_ipi)
]

print('Number of courses: ', mi_eng['course_id'].count())
#The most engaging Musical Instruments courses rows
mi_eng
wd_eng = udemy.loc[
    (udemy['subject'] == 'Web Development') &
    (udemy['price'] <= wd_mean_price) & 
    (udemy['content_duration'] >= wd_mean_con_dur) &
    (udemy['public_interest'] >= wd_mean_ipi)
]

print('Number of courses: ', wd_eng['course_id'].count())
#The most engaging Web Development courses rows
wd_eng
data = [
    [wd_mean_price, wd_mean_con_dur, wd_mean_ipi, wd_eng['course_id'].count()],
    [gd_mean_price, gd_mean_con_dur, gd_mean_ipi, gd_eng['course_id'].count()],
    [bf_mean_price, bf_mean_con_dur, bf_mean_ipi, bf_eng['course_id'].count()],
    [mi_mean_price, mi_mean_con_dur, mi_mean_ipi, mi_eng['course_id'].count()]
]

index = ['Web Development', 'Graphic Design', 'Business Finance', 'Musical Instruments']
columns = ['Mean price', 'Content duration', 'Public interest', 'Engaging courses']

total_df = pd.DataFrame(data = data, index = index, columns = columns)
total_df
paid_udemy = udemy.loc[
    udemy['is_paid'] == True
].copy()
paid_udemy.sample(5)
# <Find the content duration in minutes>
paid_udemy['contdur_min'] = paid_udemy['content_duration'] * 60
# <Find the price for 1 minute of content>
paid_udemy['price/min'] = paid_udemy['price'] / paid_udemy['contdur_min']
paid_udemy.sort_values('price/min').head()
paid_udemy[
    ['course_id', 'course_title', 'price', 'content_duration', 'price/min','subject']
].sort_values(by = 'price/min').head() 
free_udemy = udemy.loc[
    udemy['is_paid'] == False
]

free_udemy[['course_id', 'course_title', 'price','num_subscribers', 'num_reviews', 'public_interest', 'content_duration', 'subject']].sort_values('public_interest', ascending = False).head(3)
subjects = 'Business Finance', 'Web Development', 'Graphic Design', 'Musical Instruments'
amounts = [
    udemy[udemy['subject']=='Business Finance']['course_id'].count(),
    udemy[udemy['subject']=='Web Development']['course_id'].count(),
    udemy[udemy['subject']=='Graphic Design']['course_id'].count(),
    udemy[udemy['subject']=='Musical Instruments']['course_id'].count()
]
pie_chart, axes = plt.subplots()
axes.pie(amounts, labels = subjects, autopct='%.2f%%', shadow=True, radius=1.8, startangle = 45)

centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
udemy['level'].unique()
levels = 'All Levels', 'Beginner Level','Intermediate Level', 'Expert Level'
values = [
    udemy[udemy['level']=='All Levels']['course_id'].count(),
    udemy[udemy['level']=='Beginner Level']['course_id'].count(),
    udemy[udemy['level']=='Intermediate Level']['course_id'].count(),
    udemy[udemy['level']=='Expert Level']['course_id'].count()
]
bar_chart, axes = plt.subplots(figsize=(8,7))
axes.bar(levels, values)
axes.set_title('The number of courses of each subject by level')
# <Convert the column 'published_timestamp' with string values to timestamp and 
# add new column with years only>
udemy['published_timestamp'] = pd.to_datetime(udemy['published_timestamp'])
udemy['published_year'] = udemy['published_timestamp'].dt.year
udemy['published_year'].unique()
y2011 = udemy[udemy['published_year'] == 2011]['course_id'].count()
y2012 = udemy[udemy['published_year'] == 2012]['course_id'].count()
y2013 = udemy[udemy['published_year'] == 2013]['course_id'].count()
y2014 = udemy[udemy['published_year'] == 2014]['course_id'].count()
y2015 = udemy[udemy['published_year'] == 2015]['course_id'].count()
y2016 = udemy[udemy['published_year'] == 2016]['course_id'].count()
y2017 = udemy[udemy['published_year'] == 2017]['course_id'].count()
# <Data for a future graph>
years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
courses_amount = [y2011, y2012, y2013, y2014, y2015, y2016, y2017]
# <Create a function to count annual amount of published courses per subjects>
def count_courses_per_years(df, subject, year):
    amount = df[(df['subject'] == subject) & (df['published_year'] == year)]['course_id'].count()
    return amount
# <Amount of Web Development courses were published in each year>
wd_2011 = count_courses_per_years(udemy, 'Web Development', 2011)
wd_2012 = count_courses_per_years(udemy, 'Web Development', 2012)
wd_2013 = count_courses_per_years(udemy, 'Web Development', 2013)
wd_2014 = count_courses_per_years(udemy, 'Web Development', 2014)
wd_2015 = count_courses_per_years(udemy, 'Web Development', 2015)
wd_2016 = count_courses_per_years(udemy, 'Web Development', 2016)
wd_2017 = count_courses_per_years(udemy, 'Web Development', 2017)
# <Amount of Business Finance courses were published in each year>
bf_2011 = count_courses_per_years(udemy, 'Business Finance', 2011)
bf_2012 = count_courses_per_years(udemy, 'Business Finance', 2012)
bf_2013 = count_courses_per_years(udemy, 'Business Finance', 2013)
bf_2014 = count_courses_per_years(udemy, 'Business Finance', 2014)
bf_2015 = count_courses_per_years(udemy, 'Business Finance', 2015)
bf_2016 = count_courses_per_years(udemy, 'Business Finance', 2016)
bf_2017 = count_courses_per_years(udemy, 'Business Finance', 2017)
# <Amount of Graphic Design courses were published in each year>
gd_2011 = count_courses_per_years(udemy, 'Graphic Design', 2011)
gd_2012 = count_courses_per_years(udemy, 'Graphic Design', 2012)
gd_2013 = count_courses_per_years(udemy, 'Graphic Design', 2013)
gd_2014 = count_courses_per_years(udemy, 'Graphic Design', 2014)
gd_2015 = count_courses_per_years(udemy, 'Graphic Design', 2015)
gd_2016 = count_courses_per_years(udemy, 'Graphic Design', 2016)
gd_2017 = count_courses_per_years(udemy, 'Graphic Design', 2017)
# <Amount of Musical Instruments courses were published in each year>
mi_2011 = count_courses_per_years(udemy, 'Musical Instruments', 2011)
mi_2012 = count_courses_per_years(udemy, 'Musical Instruments', 2012)
mi_2013 = count_courses_per_years(udemy, 'Musical Instruments', 2013)
mi_2014 = count_courses_per_years(udemy, 'Musical Instruments', 2014)
mi_2015 = count_courses_per_years(udemy, 'Musical Instruments', 2015)
mi_2016 = count_courses_per_years(udemy, 'Musical Instruments', 2016)
mi_2017 = count_courses_per_years(udemy, 'Musical Instruments', 2017)
graph_data = [
    [wd_2011, wd_2012, wd_2013, wd_2014, wd_2015, wd_2016, wd_2017],
    [bf_2011, bf_2012, bf_2013, bf_2014, bf_2015, bf_2016, bf_2017],
    [mi_2011, mi_2012, mi_2013, mi_2014, mi_2015, mi_2016, mi_2017],
    [gd_2011, gd_2012, gd_2013, gd_2014, gd_2015, gd_2016, gd_2017],
]
annual_increase = plt.figure()
axes1 = annual_increase.add_axes([0, 0, 1.6, 0.9])
axes2 = annual_increase.add_axes([0, 1.1, 1.6, 0.5])

axes2.plot(years, courses_amount, lw = 3, marker = 'o')
axes1.plot(years, graph_data[0], label = 'Web Development', lw = 3, marker = 'o')
axes1.plot(years, graph_data[1], label = 'Business Finance', lw = 3, marker = 'o')
axes1.plot(years, graph_data[2], label = 'Musical Instruments', lw = 3, marker = 'o')
axes1.plot(years, graph_data[3], label = 'Graphic Design', lw = 3, marker = 'o')
axes1.legend()
axes1.set_title("Annual increase in subject courses")
axes2.set_title('General annual increase in courses')
udemy['published_timestamp'].sort_values(ascending = False).head(1)
udemy['published_timestamp'].sort_values().head(1)
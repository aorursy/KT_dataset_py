##Categorizing the season column using Dummy Vars : the reason is because there is no Hierarchy..

#meaning that, "Fall IS NOT Higher or Better than Summer"



def data_prep(df_clean):

    

    def parse_time(x):

        DD=datetime.strptime(x,"%m/%d/%y %H:%M")

        time=DD.hour 

        day=DD.day

        month=DD.month

        year=DD.year

        mins=DD.minute

        return time,day,month,year,mins

    

    

    

    parsed = np.array([parse_time(x) for x in df_clean.Dates])

    

    df_clean['Dates'] = pd.to_datetime(df_clean['Dates'])

    df_clean['WeekOfYear'] = df_clean['Dates'].dt.weekofyear

    #df_clean['n_days'] = (df_clean['Dates'] - df_clean['Dates'].min()).apply(lambda x: x.days)

    df_clean['HOUR'] = parsed[:,[0]]

    df_clean['day'] = parsed[:,[1]]

    df_clean['month'] = parsed[:,[2]]

    df_clean['year'] = parsed[:,[3]]

    df_clean['mins'] = parsed[:,[4]]

    

    

    #adding season variable

    def get_season(x):

        if x in [5, 6, 7]:

            r = 'summer'

        elif x in [8, 9, 10]:

            r = 'fall'

        elif x in [11, 12, 1]:

            r = 'winter'

        elif x in [2, 3, 4]:

            r = 'spring'

        return r

    

    df_clean['season'] = [get_season(i) for i in df_clean.month] 

    

    

    df_clean['Block'] = df_clean['Address'].str.contains('block', case=False)

    df_clean['Block'] = df_clean['Block'].map(lambda x: 1 if  x == True else 0)

    

    #creating dummy variables

    df_clean_onehot = pd.get_dummies(df_clean, columns=['season'], prefix = [''])

    s = (len(list(df_clean_onehot.columns))-len(df_clean.season.value_counts()))

    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,s:]], axis=1)



    ##Categorizing the DayOFWeek column using Dummy Vars 

    df_clean_onehot = pd.get_dummies(df_clean, columns=['DayOfWeek'], prefix = [''])

    

    l = (len(list(df_clean_onehot.columns))-len(df_clean.DayOfWeek.value_counts()))

    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,l:]],axis=1)



    ##Categorizing the MONTH column using Dummy Vars : the reason is because there is no Hierarchy..

    #meaning that, "FEB IS NOT Higher or Better than JAN"

    #This insight was shown from the EDA result (forecasting data with trend might be a different case)



    df_clean_onehot = pd.get_dummies(df_clean, columns=['month'], prefix = ['month'])

    n = (len(list(df_clean_onehot.columns))-len(df_clean.month.value_counts()))

    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,n:]],axis=1)



    ##Categorizing the District column using Dummy Vars 

    df_clean_onehot = pd.get_dummies(df_clean, columns=['PdDistrict'], prefix = [''])

    o = (len(list(df_clean_onehot.columns))-len(df_clean.PdDistrict.value_counts()))

    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,o:]],axis=1)

    

    df_clean['IsInterection']=df_clean['Address'].apply(lambda x: 1 if "/" in x else 0)

    df_clean['Awake']=df_clean['HOUR'].apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)

    

    ##changing the Output Variables to integer

    labels = df_clean['Category'].astype('category').cat.categories.tolist()

    replace_with_int = {'Category' : {k: v for k,v in zip(labels,list(range(0,len(labels))))}}

    df_clean.replace(replace_with_int, inplace=True)

    

    #Normalizing the columns

    def norm_func(i):

        r = (i-min(i))/(max(i)-min(i))

        return(r)



    df_clean['normHOUR']=norm_func(df_clean.HOUR)

    df_clean['normmins']=norm_func(df_clean.mins)

    df_clean['normdate_day']=norm_func(df_clean.day)

    df_clean['normLat']=norm_func(df_clean.X)

    df_clean['normLong']=norm_func(df_clean.Y)

    df_clean['normmonth']=norm_func(df_clean.month)

    df_clean['normyear']=norm_func(df_clean.year)

    df_clean['normWeekOfYear']=norm_func(df_clean.WeekOfYear)

    #df_clean['normNDAYS']=norm_func(df_clean.n_days)

    





    ##removing the unused columns

    df_clean.drop(columns = ['Dates','season','HOUR','day','X','Y'

                             ,'DayOfWeek','Address','PdDistrict','mins','month','year','WeekOfYear','Resolution'], axis = 1,inplace=True)

                             #'Count_rec_x','Count_rec_y'], axis = 1,inplace=True)

    return(df_clean)
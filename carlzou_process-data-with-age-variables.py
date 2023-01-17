def process_data(df, feature_cols,train = True):

    def process_age(df):

        #calculate mean

        ################need help with the floats###############

        df1 = df[df['age']!='1.75']

        index=df1['age'].str.find('-',0)

        index=index<=0

        mu=str(round(df1.loc[index,'age'].astype(int).mean(),0))

        

        # fill na with the mean values

        df['age'].fillna(value= mu,inplace=True)

        

        # Get the index for all age values that are not numerics, then split those at "-" and store in temp series

        inx1 = ~ df['age'].apply(lambda x: x.isnumeric())

        temp = df.loc[inx1,'age'].str.split('-')

        temp_max= df.loc[inx1,'age'].str.split('-')

        temp_min= df.loc[inx1,'age'].str.split('-')

        temp_range= df.loc[inx1,'age'].str.split('-')

        # loop over splits:

        # temp: mid point of the two values

        # temp_max: higher value of two values

        # temp_min: lower value of two values

        # temp_range: if range value then = 1, else = 0

        

        for inx2,ls in enumerate(temp):

            if len(ls) == 1:

                temp.iloc[inx2] = str(round(float(ls[0])))

                temp_max.iloc[inx2]=str(round(float(ls[0])))

                temp_min.iloc[inx2]=str(round(float(ls[0])))

                temp_range.iloc[inx2]=0

            if len(ls) == 2:

                temp.iloc[inx2] = str(round(float(((int(ls[0]) + int(ls[1]))/2))))

                temp_min.iloc[inx2] = str(round(float((int(ls[0])))))

                temp_max.iloc[inx2] = str(round(float((int(ls[1]))))) 

                temp_range.iloc[inx2]=1



        #missing values

        missing=df['age'].isnull().astype(int)



        #create age_min, age_max, age_missing, age_range

        df['age_min']=df['age']

        df['age_max']=df['age']

        df['age_missing'] = missing

        df['age_range'] = 0

        

        df.loc[inx1,'age'] = temp

        df.loc[inx1,'age_min'] = temp_min

        df.loc[inx1,'age_max'] = temp_max

        df.loc[inx1,'age_range'] = temp_range

      

      

        return df

    

    def process_symptoms(df):

        # takes symptoms column and seperates each symptom into its own column

        

        # replace NaN with Asymptomatic

        df['symptoms'].fillna('Asymptomatic',inplace=True)

        

        #split symptoms on ";"

        splits = df['symptoms'].str.split(';').values

        

        # extract array into list of list

        listOfLists = [split for split in splits]

        

        # Get unique list of all symptoms 

        symptoms = list(set(list(itertools.chain.from_iterable(listOfLists))))

        

        # create array of zeros for one hot encoding

        holder = np.zeros((len(df),len(symptoms)))

        

        # Go through dataframe and record individual symptoms

        for inx,person in enumerate(df['symptoms']):

            symptoms_ls = person.split(';')

            for symptom in symptoms_ls:

                ls_index = symptoms.index(symptom)

                holder[inx,ls_index] = 1

                

        temp_df = pd.DataFrame(data = holder, columns = symptoms)

        

        # Add temp_df to main df

        df = pd.concat([df, temp_df],axis=1)

        

        # Drop the old symptoms column (commented for now to bug-check)

        df.drop('symptoms', inplace=True, axis=1)

        

        return df

    def symptom_combiner(df, symptoms_to_combine, newcol_name):

        '''

        Combines passed symptoms into a single column named accordingly

        Assumes that the symptoms have been one hot encoded previously

        '''

        temp_df = df.loc[:,symptoms_to_combine]

        temp_df['sum'] = temp_df.sum(axis=1)

    

        sum_col= (temp_df['sum'] >=1).apply(int)



        df.drop(symptoms_to_combine, inplace=True, axis=1)

        df[newcol_name] = sum_col

        

        return df



    # Extract the feature cols

    df = df.loc[:, feature_cols]

    

    # Process age feature

    df = process_age(df)

        

    # Process symptoms

    df = process_symptoms(df)

    

#     # Process weakness

#     symptoms_to_combine = [' weakness',' malaise',' weak',' fatigue','fatigue','systemic weakness']

#     newcol_name = 'weakness'

#     df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

#     # Process coughs

#     symptoms_to_combine = [' cough','cough','dry cough']

#     newcol_name = 'cough'

#     df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

#     # process sore throat

#     symptoms_to_combine = [' throat discomfort', 'Sore throat',' sore throat','sore throat']

#     newcol_name = 'sore throat'

#     df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

#     # process diarrhea

#     symptoms_to_combine = [' diarrhoea','diarrhea',' diarrheoa']

#     newcol_name = 'diarrhea'

#     df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

#     # process nausea

#     symptoms_to_combine = [' nausea','nausea']

#     newcol_name = 'nausea'

#     df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

# #     # process pneumonia

#     symptoms_to_combine = [' severe pneumonia','pneumonitis','pneumonia']

#     newcol_name = 'pneumonia'

#     df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

#     # process fever

#     symptoms_to_combine = [' fever','fever (38-39 ° C)','low fever 37.0 ℃','fever 37.7℃','fever (39.5 ℃)',

#                            'low fever (37.2 ° C)','fever','chills']

#     newcol_name = 'fever'

#     df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

#     # process headache

#     symptoms_to_combine = [' headache','headache']

#     newcol_name = 'headache'

#     df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

#     # Process discomfort

#     symptoms_to_combine = ['chest distress','chest pain',' muscular soreness','discomfort','muscular soreness',

#                           'physical discomfort',' muscle ache',' muscular stiffness','myalgia',' myalgia']

#     newcol_name = 'discomfort'

#     df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

#     # Process shortness of breath

#     symptoms_to_combine = [' shortness of breath','severe dyspnea',' shortness breath']

#     newcol_name = 'dyspnea'

#     df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

#     # process respiratory symptoms

#     symptoms_to_combine = [' respiratory symptoms','respiratory symptoms']

#     newcol_name = 'respiratory symptoms'

#     df = symptom_combiner(df, symptoms_to_combine, newcol_name )

        

    

    # One-hot-encoding of all string features



    return df
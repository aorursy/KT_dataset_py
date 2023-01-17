def process_data(df, feature_cols):

    def process_age(df):

        # Temporarily fill na values with an absurd int

        df['age'].fillna(value= '200',inplace=True)

        

        # Get the index for all age values that are not numerics, then split those at "-" and store in temp series

        inx1 = ~ df['age'].apply(lambda x: x.isnumeric())

        temp = df.loc[inx1,'age'].str.split('-')

        

        # loop over splits, if single convert to str int, if 2 values then calculate midpoint.

        for inx2,ls in enumerate(temp):

            if len(ls) == 1:

                temp.iloc[inx2] = str(round(float(ls[0])))

            if len(ls) == 2:

                temp.iloc[inx2] = str(round(float(((int(ls[0]) + int(ls[1]))/2))))

        # replace the original age values with the processed values

        df.loc[inx1,'age'] = temp

        

        # Now deal with the NA values, replace with the group mean

        inx1 = df.loc[:,'age']=='200'

        df.loc[inx1,'age'] = df.loc[inx1,'age'] = round(df.loc[~inx1,"age"].astype(int).mean(),0)

        

        # Now replace strings with ints for better memory allocation

        df.loc[:,'age'] = df.loc[:,'age'].apply(int)

        

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

#         df.drop('symptoms', inplace=True, axis=1)

        

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

    

    # Process symptoms combine

    symptoms_to_combine = [' weakness','weakness',' weak',' fatigue','fatigue','systemic weakness']

    newcol_name = 'weakness'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

    # Process coughs

    symptoms_to_combine = [' cough','cough','dry cough']

    newcol_name = 'cough'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

    # process throat problems

    symptoms_to_combine = [' throat discomfort', 'Sore throat',' sore throat','sore throat','acute pharyngitis','pharyngeal discomfort','Pharyngeal dryness','pharynx',' pharyngeal discomfort' ]

    newcol_name = 'throat problems'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

    #process runny nose

    symptoms_to_combine = [' rhinorrhoea', ' runny nose']

    newcol_name = 'runny nose'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

    #process Chest problems

    symptoms_to_combine = ['chest tightness', 'chest distress',' pleuritic chest pain',' pleural effusion']

    newcol_name = 'Chest problem'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

     #process diarrhea

    symptoms_to_combine = [' diarrheoa', ' diarrhea', 'diarrhea']

    newcol_name = 'diarrhea'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )



    #process sore body

    symptoms_to_combine = [' muscular soreness', ' muscle ache', ' myalgia', ' soreness', ' sore body']

    newcol_name = 'sore body'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )



    #process fever

    symptoms_to_combine = ['low fever (37.4 ℃)', 'fever (38-39 ° C)','fever 37.7℃',' fever (38-39 ℃)',' fever','fever 38.3','fever',' fever (37 ℃)','fever (39.5 ℃)','low fever 37.0 ℃']

    newcol_name = 'fever'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )



    #process sputum

    symptoms_to_combine = [' expectoration', ' sputum']

    newcol_name = 'sputum'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )



    #process discomfort

    symptoms_to_combine = ['discomfort', 'malaise' ,' malaise']

    newcol_name = 'discomfort'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )

   

    #process nausea

    symptoms_to_combine = ['nausea',' nausea']

    newcol_name = 'nausea'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

    #process breathing problem

    symptoms_to_combine = [' difficulty breathing', 'anhelation', ' shortness of breath']

    newcol_name = 'breathing problem'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

    #process pneumonia

    symptoms_to_combine = ['pneumonitis', ' pneumonia', ' severe pneumonia']

    newcol_name = 'pneumonia'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

    #respiratory symptoms

    symptoms_to_combine = ['respiratory symptoms', ' respiratory symptoms']

    newcol_name = 'respiratory symptoms'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

  

    #cold

    symptoms_to_combine = [ 'chills', ' cold', ' sneezing']

    newcol_name = 'cold'

    df = symptom_combiner(df, symptoms_to_combine, newcol_name )

    

    # One-hot-encoding of all string features

#     ohe = ce.one_hot.OneHotEncoder()

#     df = ohe.fit_transform(df) # Ignore warning

    return df
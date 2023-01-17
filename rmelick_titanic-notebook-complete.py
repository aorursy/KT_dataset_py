def extract_features(rawData):

    return pd.concat([gender_features(rawData),

                     age_features(rawData)],

                    axis = 1)

    

def gender_features(rawData):

    # gender is categorical male, female, or unknown

    rawGender = pd.get_dummies(data = rawData['Sex'], dummy_na = True, prefix = 'gender_')

    return rawGender



def age_features(rawData):

    # age is continuous, but there is some missing data

    # we replace all the missing data with 0 (which means it will be)

    return rawData['Age'].fillna(0)
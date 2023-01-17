import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from transliterate import translit, get_available_language_codes

###### Function to input file name with Attributes in Russian and Translate them into Eglish for Train and Test data file format ####
def translate_data(input_file,output_file):
    full_input_file = "../input/"+input_file
    file_data = pd.read_csv(full_input_file)
    
    ##### Translate Russian language columns #####
    region = (file_data['region']).apply(translit, 'ru', reversed=True)
    city = (file_data['city']).apply(translit, 'ru', reversed=True)
    parent_category_name = (file_data['parent_category_name']).apply(translit, 'ru', reversed=True)
    category_name = (file_data['category_name']).apply(translit, 'ru', reversed=True)

    param_1 = file_data['param_1'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
    param_2 = file_data['param_2'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
    param_3 = file_data['param_3'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
    title = file_data['title'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
    description = file_data['description'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True))
    
    ### Create translated data frame #######
    file_data_translated = file_data
    file_data_translated['region'] = region
    file_data_translated['city'] = city
    file_data_translated['parent_category_name'] = parent_category_name
    file_data_translated['category_name'] = category_name
    file_data_translated['param_1'] = param_1
    file_data_translated['param_2'] = param_2
    file_data_translated['param_3'] = param_3
    file_data_translated['title'] = title
    file_data_translated['description'] = description
    
    ##### Export the translated data frame into output file  #####
    return file_data_translated
    #file_data_translated = pd.read_csv("train_translated.csv")
    #file_data_translated.head(n=5)
#### Function Calls to translate Russian language description columns into English equivalents for train.csv and test.csv #####
test_translated = translate_data("../input/test.csv","test_translated.csv")
train_translated = translate_data("../input/train.csv","train_translated.csv")

#### This function is for peforming Feature Engineering for the translated train.csv and test.csv files and export transformed data into output files ####
def transform_feature_data(file_data,output_file):
    #full_input_file = "../input/"+input_file
    #file_data = pd.read_csv(full_input_file)
    
    file_data_transformed_feature = file_data
    sliced_data_for_feature_engg = file_data.iloc[:, [2,3,4,5,6,7,8,13,14]]
    sliced_data_for_feature_engg = sliced_data_for_feature_engg.apply(lambda s: s.map({k:i for i,k in enumerate(s.unique())}))
    file_data_transformed_feature.iloc[:, [2,3,4,5,6,7,8,13,14]] = sliced_data_for_feature_engg
    file_data_transformed_feature['description'] = [len(str(x).split()) if x is not None else 0 for x in file_data_transformed_feature['description']]
    file_data_transformed_feature['title'] = [len(str(x).split()) if x is not None else 0 for x in file_data_transformed_feature['title']]

    #file_data_transformed_feature.head(5)

    ##### Export the translated data frame into output file  #####
    file_data_transformed_feature.to_csv(output_file)    
#### Function Calls for feature engineering in tranlated train.csv and test.csv and create output transformed file #####
transform_feature_data(test_translated,"test_transformed.csv")
transform_feature_data(train_translated,"train_transformed.csv")
import os
import pandas as pd

def generate_data(folder, sect):

    print("Collecting data.... ", end='')
    data = []
    count = 0
    
    for file in os.listdir(folder):
        if file == sect:
            for file in os.listdir(folder + sect):
                try:
                    text = ''
                    name = file
                    myfile = open(folder+sect+'/'+file, "r")
                    text = myfile.read()
                    mylist = [name, text]
                    count +=1
                    data.append(mylist)
                except:
                    continue

    print("collected!")
    print(str(count) + " text files found in "+ sect + " folder.")
    print("Data generated")
    return (data, count)
def match_data(data_text, data_summary, count, name):
    
    print("Creating dataframe.....", end='')
    df_text = pd.DataFrame(data_text, columns = ['File', 'Text'])
    df_sum = pd.DataFrame(data_summary, columns = ['File', 'Summary'])
    print("DONE!")
    
    print("Joining dataframes.....", end='')
    df_final = pd.merge(df_text, df_sum, on='File')
    print("DONE!")
    
    df_final.to_csv(name + '.csv')
    print(name+ ".csv Saved")
    
    return df_final
directory_text = '/kaggle/input/bbc-news-summary/BBC News Summary/News Articles/'
directory_sum = '/kaggle/input/bbc-news-summary/BBC News Summary/Summaries/'

data_business, count = generate_data( directory_text, 'business')
data_business_summary, count = generate_data( directory_sum, 'business')
df_business = match_data(data_business, data_business_summary, count, "business")

data_entertainment, count = generate_data( directory_text, 'entertainment')
data_entertainment_summary, count = generate_data( directory_sum, 'entertainment')
df_entertainment = match_data(data_entertainment, data_entertainment_summary, count, "entertainment")

data_politics, count = generate_data( directory_text, 'politics')
data_politics_summary, count = generate_data( directory_sum, 'politics')
df_politics = match_data(data_politics, data_politics_summary, count, "politics")

data_sport, count = generate_data( directory_text, 'sport')
data_sport_summary, count = generate_data( directory_sum, 'sport')
df_sport = match_data(data_sport, data_sport_summary, count, "sport")

data_tech, count = generate_data( directory_text, 'tech')
data_tech_summary, count = generate_data( directory_sum, 'tech')
df_tech = match_data(data_tech, data_tech_summary, count, "tech")
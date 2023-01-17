# Prepare Stack
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ml import
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as a_score

# scrape stack
import requests
from bs4 import BeautifulSoup
class WebCrawler(object):
    """
    This class is responsible for scrape data from website.
    Make Sure You have internet connection.
    """
    def __init__(self):
        self.url = 'http://worldpopulationreview.com/countries/'
    
    # get the header elements
    def extract_header(self, links):
        table_header = []
        for item in links:
            table_header.append(item.get('data-field'))
        table_header = [item for item in table_header if item is not None]
        return table_header
    
    # get the main data from html    
    def country_population_gen(self):
        page_as_text = requests.get(self.url).text
        soup_obj = BeautifulSoup(page_as_text,'lxml')
        ct_table = soup_obj.find('table',{'class':'table table-striped'})
        links = ct_table.findAll('a')
        table_header = self.extract_header(links)
        country_list = []
        country_stat = []
        
        for item in links:
            country = ''.join(item.findAll(text=True))
            if country not in table_header:
                country_list.append(country) 

        for trs in ct_table.findAll('tr')[1:]:
            tds = trs.findAll('td')
            row = [tds[1].text, tds[2].text.replace(',', '')]
            country_stat.append(row)
            df = pd.DataFrame.from_records(country_stat)    
            df.columns = ['country', 'population']
            df['population'] = pd.to_numeric(df.population, errors='coerce')
        return df        
    
class MetricGenerator(object):
    """
    This class is generating all the aggregated data for the poster visualizaiton
    """
    def __init__(self):
        self.df = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False, nrows=2)
        self.sdf = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False)[1:]
    
    # create a map between column header and column description
    def col_map(self):
        df_s_col = list(self.df.columns)
        df_s_col_label = self.df.loc[0].tolist()
        col_map = {name:label for name, label in zip(df_s_col, df_s_col_label)}
        return col_map
    
    # remap the gender
    def gender_mapper(self, value):
        if value == 'Male':
            return 'M'
        elif value == 'Female':
            return 'F'
        else:
            return 'O'
    # get the percentage and label of genders    
    def decompose_gender(self):
        gender = self.sdf.Q1.apply(lambda x: self.gender_mapper(x))
        cross_tab = gender.value_counts()/gender.shape*100
        label = list(cross_tab.index)
        value = cross_tab.tolist()
        return label, value
    
    # get metric and frequencies for all required metrics
    def prog_lang(self):
        top_lang_df = self.sdf.groupby("Q18").size().reset_index().rename(columns={"Q18":"language", 0:"frequency"}).nlargest(5, "frequency")
        lang = top_lang_df.language.tolist()
        lang_freq = top_lang_df.frequency.tolist()
        return lang, lang_freq
    
    def ml_tool(self):
        top_ml_tool = self.sdf.groupby("Q20").size().reset_index().rename(columns={"Q20":"ml_tool", 0:"frequency"}).nlargest(5, "frequency")
        ml_tool = top_ml_tool.ml_tool.tolist()
        ml_tool_freq = top_ml_tool.frequency.tolist()
        return ml_tool, ml_tool_freq
    
    def viz_tool(self):
        top_viz_tool = self.sdf.groupby("Q22").size().reset_index().rename(columns={"Q22":"viz_tool", 0:"frequency"}).nlargest(5, "frequency")
        viz_tool = top_viz_tool.viz_tool.tolist()
        viz_tool_freq = top_viz_tool.frequency.tolist()
        return viz_tool, viz_tool_freq
    
    def applied_ml(self):
        ml_pref = self.sdf.groupby("Q10").size().reset_index().rename(columns={"Q10":"pref_type", 0:"frequency"}).nlargest(5, "frequency")
        ml_pref_type = ["Exploring", "Not Using", "Just Started", "Unknown", "Using"]
        ml_pref_freq = ml_pref.frequency.tolist()
        return ml_pref_type, ml_pref_freq
    
    # few more dirth cleaning and joining the country and population data
    def c_index(self, df):
        name_map = {'United States of America':'United States',
               'Others' : 'Others',
               'Iran, Islamic Republic of...' : 'Iran',
               'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
               'Hong Kong (S.A.R.)': 'Hong Kong',
               'Viet Nam': 'Vietnam',
               'Republic of Korea':'South Korea'}
        country_list = df.country.tolist()
        country_map = {}
        for item in self.sdf.Q3.unique().tolist():
            if item not in country_list:
                country_map[item] = name_map.get(item)
            else:
                country_map[item] = item
        country_by_survey = self.sdf.Q3.value_counts().reset_index()
        country_by_survey['country'] = country_by_survey['index'].map(country_map)
        survey_merged = pd.merge(country_by_survey, country_df, on = 'country', how= 'left')
        survey_merged["population"] = pd.to_numeric(survey_merged.population, errors='coerec')
        survey_merged['custom_index'] = survey_merged.Q3/survey_merged.population*10000000
        survey_merged = survey_merged.sort_values("custom_index", ascending=False)
        return survey_merged
class PlotMaker(object):
    """
    This class responsible for creating custom plots
    """
    # method for creating pie plot
    def pie_plot(self, axxr, gen_label, gen_value, colors):
        patches, texts, autotexts = axxr.pie(gen_value,
                                            labels=gen_label,
                                            autopct='%1.1f%%',
                                            #explode=[0.05,0.05,0.05], 
                                            startangle=0, 
                                            pctdistance=0.85,
                                            colors=colors)
        # plt.title('Survey Response Ratio, by Gender in 2018',fontsize=20)
        # Format the text labels
        for text in texts+autotexts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        for text in autotexts:
            text.set_color('white')
        # draw circle
        centre_circle = plt.Circle((0,0),0.4,fc='white')
        axxr.add_artist(centre_circle)
        return axxr
    
    # method for creating bar plot
    def bar_plot(self, axxr, label, value, color, ylabel):
        rng = len(label)
        _ = axxr.bar(range(rng),value, align='center',color=color)
        axxr.set_xticks(range(rng))
        # axxr.set_xticklabels(labels=country_list)
        axxr.set_xticklabels(label, rotation=15)
        axxr.set_ylabel(ylabel)
        # plt.xticks(range(10),top10_arrivals_countries,fontsize=18)
        return axxr
    
    # method for creating minibar plot
    def minibar_plot(self, axxr, label, value, color):
        rng = len(label)
        _ = axxr.bar(range(rng),value, align='center',color=color)
        axxr.set_xticks(range(rng))
        # axxr.set_xticklabels(labels=country_list)
        axxr.set_xticklabels(label,fontdict={'fontweight': 'bold'} ,rotation=60)
        axxr.yaxis.set_visible(False)
        # plt.xticks(range(10),top10_arrivals_countries,fontsize=18)
        return axxr
class DecisionSupportSystem(object):
    """
    This class will generate data and build the model
    """
    # remap the column for system
    def __init__(self):
        self.sdf = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False)[1:]
        self.ml_col_list = ["Q2", "Q4", "Q8", "Q37","Q17", "Q20", "Q23", "Q24","Q6"]
        self.age_map_q2 = {'18-21': 1, '22-24': 2, '25-29': 3, '30-34': 4, '35-39': 5, '40-44': 6, '45-49': 7, '50-54': 8, '55-59': 9, '60-69': 10, '70-79': 11, '80+': 12}
        self.education_map_q4 = {'Doctoral degree' : 6, 'Bachelor’s degree' : 4, 'Master’s degree' : 5,
                         'Professional degree' : 3, 'Some college/university study without earning a bachelor’s degree' : 2,
                         'I prefer not to answer' : 0, 'No formal education past high school' : 1}
        self.experience_map_q8 = {'0-1': 1, '1-2': 2, '2-3': 3, '3-4': 4, '4-5': 5, '5-10': 6, '10-15': 7, '15-20': 8, '20-25': 9, '25-30': 10, "30 +": 11}
        self.course_map_q37 = {'DataCamp': 'datac', 'Coursera': 'cours', 'Kaggle Learn': 'kaggl', 'Udacity': 'udaci', 
                       'Other': 'other', 'developers.google.com': 'devel', 'Online University Courses': 'onlin', 
                       'Udemy': 'udemy', 'DataQuest': 'dataq', 'Fast.AI': 'fasta', 'TheSchool.AI': 'thesc'}
        self.job_map_q6 = {'Consultant': 'consu', 'Data Scientist': 'datas', 'Data Analyst': 'dataa', 'Software Engineer': 'softw', 
                   'Research Assistant': 'resea', 'Chief Officer': 'chief', 'Research Scientist': 'resea', 
                   'Business Analyst': 'busin', 'Data Engineer': 'datae', 'Developer Advocate': 'devel', 
                   'Marketing Analyst': 'marke', 'Product/Project Manager': 'produ', 'Principal Investigator': 'princ', 
                   'DBA/Database Engineer': 'dbada', 'Statistician': 'stati', 'Data Journalist': 'dataj'}
        
        self.reverse_job_map = {'consu': 'Consultant', 'datas': 'Data Scientist', 'dataa': 'Data Analyst', 
                           'softw': 'Software Engineer','resea': 'Research Scientist', 'chief': 'Chief Officer', 
                           'busin': 'Business Analyst', 'datae': 'Data Engineer', 'devel': 'Developer Advocate', 
                           'marke': 'Marketing Analyst', 'produ': 'Product/Project Manager', 
                           'princ': 'Principal Investigator', 'dbada': 'DBA/Database Engineer', 
                           'stati': 'Statistician', 'dataj': 'Data Journalist'}

        self.job_map_q6b = {'Consultant': 'consu', 'Data Scientist': 'datas', 'Data Analyst': 'dataa',
                       'Research Assistant': 'resea', 'Research Scientist': 'resea', 'Business Analyst': 'busin', 
                       'Data Engineer': 'datae', 'Marketing Analyst': 'marke', 'Statistician': 'stati'}

        self.lang_map_q17 = {'Java': 'java', 'Python': 'python', 'SQL': 'sql', 'Javascript/Typescript': 'javascript/typescript', 
                        'C#/.NET': 'c#/.net', 'R': 'r', 'MATLAB': 'matlab', 'C/C++': 'c/c++', 'Visual Basic/VBA': 'visual_basic/vba', 
                        'Bash': 'bash', 'Scala': 'scala', 'PHP': 'php', 'SAS/STATA': 'sas/stata', 'Other': 'other', 'Ruby': 'ruby', 
                        'Go': 'go', 'Julia': 'julia'}
        self.code_time_map_q23 = {'0% of my time' : 1, '1% to 25% of my time': 2, '75% to 99% of my time' : 5, 
                         '50% to 74% of my time': 4, '25% to 49% of my time' : 3,'100% of my time': 6}
        self.code_exp_map_q24 = {'I have never written code but I want to learn' : 2, '5-10 years' : 6,
                        '3-5 years' : 5, '< 1 year' : 3, '1-2 years' : 4, '10-20 years': 7, '20-30 years': 8,
                        '30-40 years': 9,'I have never written code and I do not want to learn' : 1, '40+ years': 10}
        self.ml_pak_map_q20 = {'Scikit-Learn': 'scikit-learn', 'Keras': 'keras', 'TensorFlow': 'tensorflow', 'Caret': 'caret', 
                      'Spark MLlib': 'spark-mllib', 'Caffe': 'caffe', 'mlr': 'mlr', 'PyTorch': 'pytorch', 
                      'randomForest': 'randomforest', 'H20': 'h20', 'Xgboost': 'xgboost', 'lightgbm': 'lightgbm', 
                      'Fastai': 'fastai', 'Other': 'other', 'CNTK': 'cntk', 'catboost': 'catboost', 
                      'Prophet': 'prophet', 'Mxnet': 'mxnet'}
        
    # method to get the reverse map from the previous map
    def reverse_map(self, dict_obj):
        reverse_map = {value:key for key, value in dict_obj.items()}
        return reverse_map
    # method for one vs all encoding for response variable
    def one_all_encode(self, response, job_code):
        y_train = response.apply(lambda x: 1 if x==job_code else 0)
        return y_train
    
    def prepare_ml_data(self, job_code):
        # impute and clean the data for model
        ml_df = self.sdf[self.ml_col_list]
        ml_df =  ml_df.dropna(axis=0, subset=['Q6', 'Q8'])
        ml_df["Q6"] = ml_df.Q6.apply(lambda x: np.nan if x is np.nan else self.job_map_q6b.get(x, np.nan))
        ml_df =  ml_df.dropna(axis=0, subset=['Q6'])
        ml_df["Q2"] = ml_df.Q2.map(self.age_map_q2)
        ml_df["Q4"] = ml_df.Q4.apply(lambda x: self.education_map_q4.get(x))
        ml_df["Q8"] = ml_df.Q8.apply(lambda x: 0 if x is np.nan else self.experience_map_q8.get(x))
        ml_df["Q37"] = ml_df.Q37.apply(lambda x: "unkno37" if x is np.nan else self.course_map_q37.get(x))
        ml_df["Q17"] = ml_df.Q17.apply(lambda x: "unkno17" if x is np.nan else self.lang_map_q17.get(x))
        ml_df["Q20"] = ml_df.Q20.apply(lambda x: "unkno20" if x is np.nan else self.ml_pak_map_q20.get(x))
        ml_df["Q23"] = ml_df.Q23.apply(lambda x: 0 if x is np.nan else self.code_time_map_q23.get(x))
        ml_df["Q24"] = ml_df.Q24.apply(lambda x: 0 if x is np.nan else self.code_exp_map_q24.get(x))
        ml_df = ml_df[ml_df.Q4 != 0]
        # get dummy variables
        online_platform = pd.get_dummies(ml_df.Q37, drop_first=True)
        programming = pd.get_dummies(ml_df.Q17, drop_first=True)
        ml_pack = pd.get_dummies(ml_df.Q20, drop_first=True)
        # remove rows with null value
        ml_df = ml_df.drop(["Q37", "Q17", "Q20"], axis=1)
        ml_df = pd.concat([ml_df, online_platform, programming, ml_pack], axis=1)
        # remove un necessary columns
        X_df = ml_df.drop("Q6", axis=1)
        y_series = self.one_all_encode(ml_df.Q6, job_code)
        return X_df, y_series
    
    def fit_model(self,X_set, y_set):
        # sorten the question map
        prediction_label_map = {"Q2" : "Age(year)" ,"Q4" : "Education Level" ,"Q8" : "Experience", "Q23" : "Time Spent on Coding" ,
                            "Q24" : "Coding Experience", "p_1" : "Probability", "online_platform" : "Preferred Online Platform", 
                            "prog_lang" : "Preferrend Language", "mlp_pak" : "Preferred ML Platform"}
        # create train test split
        X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.2,random_state=101)
        logmodel = LogisticRegression(C=100.0, random_state=1)
        logmodel.fit(X_train, y_train)
        predictions = logmodel.predict(X_test)
        accuracy_value = a_score(y_test, predictions)
        r_square_score = logmodel.score(X_set, y_set)
        # get the probability from the model
        prob_df = pd.DataFrame(logmodel.predict_proba(X_train), columns=["p_0", "p_1"])
        final_predict = pd.concat([X_train.reset_index(drop=True), prob_df], axis=1)
        result = final_predict.nlargest(3, "p_1").rename_axis("_id").reset_index()
        result = result.drop_duplicates(['p_1'], keep='first')
        # convert data from wide to long format
        column_platform = ['_id', 'datac', 'dataq', 'devel', 'fasta', 'kaggl', 'onlin', 'other', 'thesc', 'udaci', 'udemy','unkno37']
        column_programming = ['_id', 'c#/.net', 'c/c++', 'go', 'java', 'javascript/typescript','julia', 'matlab', 'other', 
                              'php', 'python', 'r', 'ruby', 'sas/stata', 'scala', 'sql', 'unkno17', 'visual_basic/vba']
        column_ml_pak = ['_id', 'caret', 'catboost','cntk', 'fastai', 'h20', 'keras', 'lightgbm', 'mlr', 'mxnet', 
              'other', 'prophet', 'pytorch', 'randomforest', 'scikit-learn', 'spark-mllib', 
              'tensorflow', 'unkno20', 'xgboost']
        column_rest = ['_id', 'Q2', 'Q4', 'Q8', 'Q23', 'Q24', 'p_1']
        # select the rows hiving value only 1 for hot encoding columns
        result_plt = pd.melt(result[column_platform], id_vars=['_id'], var_name="online_platform", value_name="platform_encoding")
        result_plt = result_plt[result_plt.platform_encoding == 1].set_index("_id")
        result_prg = pd.melt(result[column_programming], id_vars=['_id'], var_name="prog_lang", value_name="lang_encoding")
        result_prg = result_prg[result_prg.lang_encoding == 1].set_index("_id")
        result_mlp = pd.melt(result[column_ml_pak], id_vars=['_id'], var_name="mlp_pak", value_name="pak_encoding")
        result_mlp = result_mlp[result_mlp.pak_encoding == 1].set_index("_id")
        result_rest = result[column_rest].set_index("_id")
        result_top = result_rest.join(result_plt.drop("platform_encoding", axis=1)).join(result_prg.drop("lang_encoding", axis=1))
        result_top = result_top.join(result_mlp.drop("pak_encoding", axis=1)).reset_index(drop=True)
        # get the actual values
        result_top["Q2"] = result_top.Q2.map(self.reverse_map(self.age_map_q2))
        result_top["Q4"] = result_top.Q4.map(self.reverse_map(self.education_map_q4))
        result_top["Q8"] = result_top.Q8.map(self.reverse_map(self.experience_map_q8))
        result_top["Q23"] = result_top.Q23.map(self.reverse_map(self.code_time_map_q23))
        result_top["Q24"] = result_top.Q24.map(self.reverse_map(self.code_exp_map_q24))
        result_top["online_platform"] = result_top.online_platform.apply(lambda x: self.reverse_map(self.course_map_q37).get(x, "Not Found"))
        result_top["prog_lang"] = result_top.prog_lang.map(self.reverse_map(self.lang_map_q17))    
        result_top["mlp_pak"] = result_top.mlp_pak.apply(lambda x: self.reverse_map(self.ml_pak_map_q20).get(x, "Not Found"))  
        result_top["p_1"] = result_top.p_1.apply(lambda x: str(x)[:4])
        result_top = result_top.rename(columns=prediction_label_map)
        return result_top, accuracy_value, r_square_score
    # convert the result to string
    def result_to_string(self, result_item):
        string = ''
        for key, value in result_item.items():
            string = string + "{} : {} ".format(key, value) + "\n"
        return string
    
    # finction to create the visual poster
    def persona_poster(self, label, result_dict):
        mpl.style.use('seaborn')
        fontparams = {'size':12,'fontweight':'light', 'family':'monospace','style':'normal'}
        fig = plt.figure(figsize=(9.9,14))
        G = gridspec.GridSpec(10, 6)
        head_axes = plt.subplot(G[0, :], facecolor='teal')
        plt.xticks(())
        plt.yticks(())
        plt.text(0.5, 0.5, 'Be like {} !'.format(label), color='white', ha='center', va='center', size=24)

        foot_axes = plt.subplot(G[9, :], facecolor='teal')
        plt.xticks(())
        plt.yticks(())
        plt.text(0.5, 0.5, '© Copyright 2018, XXX \n source: https://www.kaggle.com/kaggle/kaggle-survey-2018 \n\nAuthor: DataPsycho', color='white', ha='center', va='center', size=14)

        axes_12a = plt.subplot(G[1:3, :-4], facecolor='#333333')
        plt.xticks(())
        plt.yticks(())
        plt.text(0.5, 0.5, 'Model Info.', color='white', ha='center', va='center', size=24, alpha=1.0)

        axes_12b = plt.subplot(G[1:3, 2:], facecolor='#333333')
        plt.xticks(())
        plt.yticks(())
        plt.text(0.5, 0.6, 'R Square : {:.2f} \n \n Test Accuracy: {:.2f}'.format(r_square, accuracy), color='white', ha='center', va='center', size=15, alpha=1.0, fontdict=fontparams)

        axes_34a = plt.subplot(G[3:5, :-4], facecolor='#041f33')
        plt.xticks(())
        plt.yticks(())
        plt.text(0.5, 0.5, 'Bilkis', color='white', ha='center', va='center', size=24, alpha=1.0)

        axes_34b = plt.subplot(G[3:5, 2:], facecolor='#041f33')
        plt.xticks(())
        plt.yticks(())
        plt.text(0.5, 0.5, dss.result_to_string(result_dict[0]), color='white', ha='center', va='center', size=14, alpha=1.0, fontdict=fontparams)

        axes_56a = plt.subplot(G[5:7, :-4], facecolor='#45002d')
        plt.xticks(())
        plt.yticks(())
        plt.text(0.5, 0.5, 'Mofiz', color='white', ha='center', va='center', size=24, alpha=1.0)

        axes_56b = plt.subplot(G[5:7, 2:], facecolor='#45002d')
        plt.xticks(())
        plt.yticks(())
        plt.text(0.5, 0.5, dss.result_to_string(result_dict[1]), color='white', ha='center', va='center', size=14, alpha=1.0, fontdict=fontparams)

        axes_78a = plt.subplot(G[7:9, :-4], facecolor='#03314a')
        plt.xticks(())
        plt.yticks(())
        plt.text(0.5, 0.5, 'Adéla', color='white', ha='center', va='center', size=24, alpha=1.0)

        axes_78b = plt.subplot(G[7:9, 2:], facecolor='#03314a')
        plt.xticks(())
        plt.yticks(())
        plt.text(0.5, 0.5, dss.result_to_string(result_dict[2]), color='white', ha='center', va='center', size=14, alpha=1.0, fontdict=fontparams) 
        plt.tight_layout()
        return plt.show()
crawler = WebCrawler()
metric_manager = MetricGenerator()
plot_manager = PlotMaker()
country_df = crawler.country_population_gen()
cindex_df = metric_manager.c_index(country_df)
print("Top 10 Countries According to C Index.")
print(cindex_df.drop(["index", "Q3"], axis=1).head(10))
def hightlights_poster():
    mpl.style.use('seaborn')
    fig = plt.figure(figsize=(12.7,18))
    G = gridspec.GridSpec(10, 5)
    head_axes = plt.subplot(G[0, :], facecolor='teal')
    plt.xticks(())
    plt.yticks(())
    plt.text(0.5, 0.5, 'Kaggle Survey Highlights, 2018', color='white', ha='center', va='center', size=24)

    foot_axes = plt.subplot(G[9, :], facecolor='teal')
    plt.xticks(())
    plt.yticks(())
    plt.text(0.5, 0.5, '© Copyright 2018, XXX \n source: https://www.kaggle.com/kaggle/kaggle-survey-2018 \n\nAuthor: DataPsycho', color='white', ha='center', va='center', size=14)

    axes_12a = plt.subplot(G[1:3, :-3], facecolor='lightcoral')
    plt.xticks(())
    plt.yticks(())
    plt.text(0.5, 0.5, 'Hire More Women !', color='black', ha='center', va='center', size=24, alpha=1.0)
    plt.text(0.5, 0.3, 'Approximately each 5 resopnse \nthere is only 1 woman responded, \nWomen might have less participation in Data Science.', color='black', ha='center', va='center', size=12, alpha=1.0)

    axes_12b = plt.subplot(G[1:3, 2:])
    label, value = metric_manager.decompose_gender()
    colors = ['navy','lightcoral', 'grey']
    plot_manager.pie_plot(axes_12b, label, value, colors)
    plt.text(0.0, 0.0, 'M : F \n4 : 1', color='black', ha='center', va='center', size=12)

    axes_34a = plt.subplot(G[3:5, 2:], facecolor='lightgrey')
    bottom_country = cindex_df.nsmallest(10, 'custom_index').sort_values('custom_index', ascending=False)
    country_list = bottom_country.country.tolist()
    custom_index = bottom_country.custom_index.tolist()
    plot_manager.bar_plot(axes_34a, country_list, custom_index, '#3b5998', 'C Index')

    axes_34b = plt.subplot(G[3:5, :2], facecolor='#3b5998')
    plt.xticks(())
    plt.yticks(())
    plt.text(0.5, 0.5, 'Together We can Go Far !', color='white', ha='center', va='center', size=20)
    plt.text(0.5, 0.3, 'The bottom 10 Countries according to \nC Index (responses per 10M people).', color='white', ha='center', va='center', size=12, alpha=1.0)

    axes_56a = plt.subplot(G[5:7, :-3], facecolor='#2a4b5a')
    plt.xticks(())
    plt.yticks(())
    plt.text(0.5, 0.5, 'Programmer\'s Tools!', color='white', ha='center', va='center', size=24)
    plt.text(0.5, 0.3, 'Python, Scikit-Learn, Matplot \nis the ultimate bundle.', color='white', ha='center', va='center', size=12, alpha=1.0)

    axes_56b = plt.subplot(G[5:7, 2:], facecolor='lightgrey')
    lang, lang_freq = metric_manager.prog_lang()
    plot_manager.bar_plot(axes_56b, lang, lang_freq, '#2a4b5a',  'no. of responses')
    subax1 = inset_axes(axes_56b, width=2, height=1, loc=1)
    ml_tool, ml_tool_freq = metric_manager.ml_tool()
    plot_manager.minibar_plot(subax1, ml_tool, ml_tool_freq, '#2a4b5a')
    subax2 = inset_axes(axes_56b, width=2, height=1, loc=9)
    viz_tool, viz_tool_freq = metric_manager.viz_tool()
    plot_manager.minibar_plot(subax2, viz_tool, viz_tool_freq, '#2a4b5a')

    axes_78a = plt.subplot(G[7:9, :-3], facecolor='#4a2339')
    plt.xticks(())
    plt.yticks(())
    plt.text(0.5, 0.5, 'ML: Still A Buzzword?', color='white', ha='center', va='center', size=24)
    plt.text(0.5, 0.3, 'Dawn of Applied ML implies \nproper ML culture has not been stablished yet.', color='white', ha='center', va='center', size=12, alpha=1.0)

    axes_78b = plt.subplot(G[7:9, 2:], facecolor='lightgrey')
    ml_pref_type, ml_pref_freq = metric_manager.applied_ml()
    plot_manager.bar_plot(axes_78b, ml_pref_type, ml_pref_freq, '#4a2339',  'no. of responses')
    plt.tight_layout()
    return plt.show()

hightlights_poster()
# attach the class
dss = DecisionSupportSystem()
# assign a profession
profession =  "datas"
# get the actual label
profession_label = dss.reverse_job_map.get(profession).upper()
# get the prepared data
X_df, y_series = dss.prepare_ml_data(profession)
# fit the model and get required metrics
result, accuracy, r_square = dss.fit_model(X_df, y_series)
# convert result to dict object
result_dict = result.to_dict(orient="records")
# create the poster
dss.persona_poster(profession_label, result_dict)
profession =  "dataa"
profession_label = dss.reverse_job_map.get(profession).upper()
X_df, y_series = dss.prepare_ml_data(profession)
result, accuracy, r_square = dss.fit_model(X_df, y_series)
result_dict = result.to_dict(orient="records")
result
dss.persona_poster(profession_label, result_dict)
profession =  "resea"
profession_label = dss.reverse_job_map.get(profession).upper()
X_df, y_series = dss.prepare_ml_data(profession)
result, accuracy, r_square = dss.fit_model(X_df, y_series)
result_dict = result.to_dict(orient="records")
dss.persona_poster(profession_label, result_dict)
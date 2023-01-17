from IPython.display import display, Markdown
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
sns.set_color_codes()
from os import listdir
from os.path import isfile, join

%matplotlib inline

# Ignore the warnings
import warnings
warnings.filterwarnings('ignore')

fontdict = {'fontsize':20, 'fontweight':'bold'}

class KaggleSurvey:
    
    def __init__(self):
        self.__asset_path = "../input/"
        self.__get_dfs()
        self.__save_dfs()
        self.__make_q_list()
    
    def __get_dfs(self):
        path = self.__asset_path
        dict_2018 = {}
        files = [f for f in listdir(path) if isfile(join(path, f)) and "csv" in f]
        for idx_, file in enumerate(files):
            file_path = path + file
            df = pd.read_csv(file_path, encoding="ISO-8859-1")
            dict_2018[file.split(".")[0]] = df

        self.__dict_2018 = dict_2018
        
    def __save_dfs(self):
        q_list = self.__dict_2018["multipleChoiceResponses"].iloc[0].values.tolist()
        self.__schema_new = pd.concat(
            [pd.Series(self.__dict_2018["multipleChoiceResponses"].columns), pd.Series(q_list)], 1
        )
        self.__multi_18 = self.__dict_2018["multipleChoiceResponses"].iloc[1:].reset_index(drop=True)
        self.__free_18 = self.__dict_2018["freeFormResponses"].iloc[1:].reset_index(drop=True)
        cols = self.__dict_2018["SurveySchema"].columns
        rearrange_cols = ["Q" + str(i) for i in range(len(cols))]
        rearrange_cols[0], rearrange_cols[-1] = cols[0], cols[-1]
        self.__survey_schema_18 = self.__dict_2018["SurveySchema"][rearrange_cols]
    
    def __make_q_list(self):
        single_choice, multiple_choice = self.__count_single_multiple_choices()
        q_list, is_single = self.__make_question_list(single_choice, multiple_choice)
        self.__q_df = pd.DataFrame({"question" : pd.Series(q_list), "is_single" : pd.Series(is_single)})
        self.__q_df.index = range(1, 51)
        # question 12 has unique column name so I'll mark it as single choice question even though it's not.
        self.__q_df.at[12, 'is_single'] = 1
    
    def __count_single_multiple_choices(self):
        cols = self.__multi_18.columns
        single_choice = []
        multiple_choice = []
        for col in cols[1:]:
            tmp = col.split("_")
            if len(tmp) == 1:
                single_choice.append(col)
            elif "Part" in tmp:
                if tmp[0] not in multiple_choice:
                    multiple_choice.append(tmp[0])
        return single_choice, multiple_choice
        
    def __make_question_list(self, single_choice, multiple_choice):
        tmp_df = self.__schema_new
        tmp_df.columns = ["Column", "Question"]
        tmp_df = tmp_df.set_index("Column")
        q_list = []
        q_len = len(single_choice) + len(multiple_choice)
        is_single_choice_list = []
        for i in range(q_len):
            is_single = 1
            q_txt = "Q" + str(i + 1)
            if q_txt in multiple_choice:
                q_txt = q_txt + "_Part_1"
                is_single = 0
            try:
                q = tmp_df.loc[q_txt]["Question"]
            except:
                q_txt = q_txt + "_TEXT"
                q = tmp_df.loc[q_txt]["Question"]
            q_list.append(q)
            is_single_choice_list.append(is_single)
        return q_list, is_single_choice_list

    def __get_selections_of_multiple_choice_question_as_list(self, number):
        tmp_df = self.__schema_new
        tmp_df.columns = ["Column", "Question"]
        tmp_df = tmp_df.set_index("Column")
        tmp_li = [ 
            q_ for q_ in 
            (q for q in tmp_df.index.tolist() if str(number) in q) 
            if str(number) in q_.split("_")[0]
        ]
        return tmp_li
    
    def __print_question(self, year, col, cols=[], is_survey_schema=False, is_need_result = False):
#         if is_survey_schema:
#             tmp_df = pd.read_csv(self.__asset_path + "surveySchema_18.csv")
#             print(tmp_df[col].iloc[0])
#             return 
        dir_name = "kaggle-survey-"
        tmp_df = pd.DataFrame([])
        result = ""
        tmp_df = self.__schema_new
        tmp_df.columns = ["Column", "Question"]
        tmp_df = tmp_df.set_index("Column")
        result = tmp_df.loc[col]["Question"]
        if is_need_result:
            return result
        else:
            print(col)
            if len(cols) == 0:
                print(result)
        
    def __get_order_li(self, df, q, question_number):
        return [str_ for str_ in df[q].unique().tolist() if type(str_) != float]
    
    def __get_ax(self, ax, idx, plot_cols, depth_of_dfs):
        ax_ = None
        if plot_cols == 1:
            ax_ = ax[idx] 
        elif plot_cols > 1:
            ax_ = ax[idx // plot_cols, idx % plot_cols] if depth_of_dfs > 1 else ax[idx % plot_cols]
        else:
            print("wrong plot_cols.")
        return ax_
        
    def __per_df(self, series) :
        val_cnt = series.values.sum()
        return series / val_cnt
    
    def __per_df_single(self, df, col, order_li):
        series = pd.Series(index = order_li)
        idx_li = df[col].value_counts().index.tolist()
        for idx in idx_li:
            series.at[idx] = df[col].value_counts().loc[idx]
        return series / series.sum()
    
    def get_question(self, number, is_need_display_order = False):
        """
            print the question 
            number : Insert the number of question.
        """
        df = self.__q_df
        tmp = df.loc[number]
        if tmp["is_single"] == 1:
            q = tmp["question"]
            print("Q" +str(number) + ".", df.loc[number]["question"])
        else:
            tmp_li = self.__get_selections_of_multiple_choice_question_as_list(number)
            display_order = []
            for idx, q in enumerate(tmp_li):
                if "_OTHER_TEXT" not in q:                
                    q_ = self.__print_question(2018, q, is_need_result=True).split(" - ")
                    display_order.append(q_[-1].strip())
                    if idx == 0:
                        print(q.split("_")[0] + ".", q_[0])
                        print(" ", str(idx + 1) + ".", q_[-1])
                    else:
                        print(" ", str(idx + 1) + ".", q_[-1])
            if is_need_display_order:
                return display_order
    
    def get_q_df(self):
        """
            return dataframe of survey questions, columns = ["is_single_choice", "question"]
        """
        return self.__q_df
    
    def get_multiple_choice_responses(self):
        """
            return multipleChoiceResponses dataframe
        """
        return self.__multi_18
    
    def get_free_form_responses(self):
        """
            return freeFormResponses dataframe
        """
        return self.__free_18

    def get_survey_schema(self):
        """
            return SurveySchema dataframe
        """
        return self.__survey_schema_18
    
    def get_df_saved_in_object(self):
        """
            return dictionary of dataframes saved in object
        """
        return self.__dfs
    
    def get_df_I_want_by_country_name(self, country, is_false = False):
        """
            return the dataset of certain ountry
            country : Insert the country name
        """
        df = self.__multi_18.copy()
        col = "Q3"
        countrie_names = df[col].unique().tolist()
        if country not in countrie_names:
            print("Wrong country name. You can only insert the one of names below.")
            print(countrie_names)
            return
                
        return df[df[col] != country] if is_false else df[df[col] == country]
    
    def set_df_I_want_by_country_name(self, countries = [], is_need_total = False):
        """
            To set dictionary of dataframe in the object. This way you can maintain your code more easier.
            contries : Insert the list of country names
                example:) ["Korea", "China", "India" ... ]
            is_need_total : True if you want the full dataset to be in inside of object 
        """
        
        keys = []
        values = []
        if is_need_total:
            keys.append("Total")
            values.append(self.__multi_18)
            
        countrie_names = self.__multi_18["Q3"].unique().tolist()
        for country in countries:
            if country not in countrie_names:
                print("Wrong country name. You can only insert the one of names below.")
                print(countrie_names)
                break
            df = self.get_df_I_want_by_country_name(country)
            keys.append(country)
            values.append(df)
                                
        self.__dfs = dict(zip(keys, values))

    def save_csv(self, df, filename, index=False):
        df.to_csv(self.__asset_path + filename, index = index)
        
    def draw_plot(self, question_number, plot_cols = 3, dfs_ = {}, is_need_number_of_respondents = False, order = [], 
                  is_use_the_same_y_lim = True, ylim_offset = 0.1):
        """
            question_number : Insert the number of question
            plot_cols : Insert the number of plot's columns, default 3
            dfs_ : Insert the dictionary of dataframes.
                Example:) dfs_ = {
                    "US" : ks.get_df_I_want_by_country_name('United States of America'),
                    'India' : ks.get_df_I_want_by_country_name('India'),
                    'China' : ks.get_df_I_want_by_country_name('China')
                }
            order : Insert the order of xticks.
            is_use_the_same_y_lim : True
            ylim_offset : 0.1 (it means 10%)
        """
        
        dfs_keys = []
        dfs = []
        if len(dfs_) != 0:
            dfs_keys = list(dfs_.keys())
            dfs = dfs_.values()
        elif len(self.__dfs) != 0:
            dfs_keys = list(self.__dfs.keys())
            dfs = self.__dfs.values()
        else:
            print("Insert the dataset or set the dataset into object using set_df_I_want_by_country_name")
        
        is_single_choice_question = self.__q_df.loc[question_number]["is_single"]
        
        length_of_dfs = len(dfs)
        depth_of_dfs = int(np.ceil(length_of_dfs / float(plot_cols)))
        if question_number != 3:
            f, ax = plt.subplots(depth_of_dfs, plot_cols, figsize=(5 * length_of_dfs, 5 * depth_of_dfs))
        
        if is_single_choice_question:
            if question_number == 12:
                display(Markdown("<span style='color: red'>Multiple Choice Question.</span>"))
            else:
                display(Markdown("<span style='color: red'>Single Choice Question.</span>"))
            self.get_question(question_number)
            if question_number == 3:
                len_li = [len(df) for df in dfs]
                plt.bar(dfs_keys, len_li)
                
            else:        
                q = "Q" + str(question_number) + "_MULTIPLE_CHOICE" if question_number == 12 else "Q" + str(question_number)


                if is_use_the_same_y_lim:
                    max_value = 0
                    for df in dfs:
                        ncount = len(df)
                        tmp = df.groupby(q).size().values/ncount
                        tmp_max_value = (tmp.max() + ylim_offset)
                        max_value = tmp_max_value if tmp_max_value > max_value else max_value
                
                order_li = []
                print("The answers from left to right : ")
                
                for idx, df in enumerate(dfs):
                    
                    if idx == 0:
                        if len(order) != 0:
                            order_li = order
                        else:
                            order_li = self.__get_order_li(self.__multi_18, q, question_number)
                    
                    ax_ = self.__get_ax(ax, idx, plot_cols, depth_of_dfs)
                    
                    self.__per_df_single(df, q, order_li).plot.bar(ax = ax_)
                    
                    for p in ax_.patches:
                        ax_.annotate(str(round(p.get_height() * 100, 2)) + "%", (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
                        
                    ax_.set_xticklabels(list(range(1, len(order_li) + 1)), rotation=0)
                    if is_need_number_of_respondents:
                        ax_.set_title(dfs_keys[idx] + " (" + str(len(df)) + ")", fontdict = fontdict)
                    else:
                        ax_.set_title(dfs_keys[idx], fontdict = fontdict)

                    if is_use_the_same_y_lim:
                        ax_.set_ylim(0, max_value)
                        
                for idx, answer in enumerate(order_li):
                    print(idx + 1, answer)
                plt.show()
                
        else:
            display(Markdown("<span style='color: red'>Multiple Choice Question.</span>"))
            answers = self.get_question(question_number, is_need_display_order=True)
            cols = [
                q for q in self.__get_selections_of_multiple_choice_question_as_list(question_number) if "_OTHER_TEXT" not in q
            ] 
            order_li = []
            result_dfs = []
            # 34, 35 answers must add up to 100%
            if question_number == 34 or question_number == 35:
                if is_use_the_same_y_lim:
                    max_value = 0
                    for df in dfs:
                        for col in cols:
                            mean_max = df[cols].astype(float).mean().max()
                            tmp_max_value = (mean_max + ylim_offset * 100)
                            max_value = tmp_max_value if tmp_max_value > max_value else max_value
                
                for idx, df in enumerate(dfs):
                    ax_ = self.__get_ax(ax, idx, plot_cols, depth_of_dfs)
                    sns.barplot(data = df[cols], ax = ax_)
                    ax_.set_xticklabels(list(range(1, len(df.columns) + 1)), rotation = 0)
                    if is_need_number_of_respondents:
                        ax_.set_title(dfs_keys[idx] + " (" + str(len(df[cols])) + ")", fontdict = fontdict)
                    else:
                        ax_.set_title(dfs_keys[idx], fontdict = fontdict)
                    if is_use_the_same_y_lim:
                        ax_.set_ylim(0, max_value)
            else:
                for idx, df in enumerate(dfs):
                    x_li = []
                    y_li = []
                    for i, col in enumerate(cols):
                        uq_li = df[col].unique().tolist()
                        if len(uq_li) > 1:
                            if str(uq_li[0]) == "nan":
                                y_li.append(uq_li[1])
                                x_li.append(len(df[df[col] == uq_li[1]]))
                            else:
                                y_li.append(uq_li[0])
                                x_li.append(len(df[df[col] == uq_li[0]]))
                        else:
                            y_li.append(answers[i])
                            x_li.append(0)
                    result_dfs.append(pd.Series(x_li, y_li))
                    if len(order_li) == 0:
                        order_li = y_li

                if is_use_the_same_y_lim:
                    max_value = 0
                    for df in result_dfs:
                        tmp = df/df.sum()
                        tmp_max_value = (tmp.max() + ylim_offset)
                        max_value = tmp_max_value if tmp_max_value > max_value else max_value

                for idx, df in enumerate(result_dfs):
                    if idx == 0:
                        if len(order) != 0:
                            order_li = order
                    ax_ = self.__get_ax(ax, idx, plot_cols, depth_of_dfs)
                    self.__per_df(df).plot.bar(ax = ax_)
                    for p in ax_.patches:
                        ax_.annotate(str(round(p.get_height() * 100, 2)) + "%", (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
                    ax_.set_xticklabels(list(range(1, len(order_li) + 1)), rotation = 0)
                    if is_need_number_of_respondents:
                        ax_.set_title(dfs_keys[idx] + " (" + str(df.values.sum()) + ")", fontdict = fontdict)
                    else:
                        ax_.set_title(dfs_keys[idx], fontdict = fontdict)
                    if is_use_the_same_y_lim:
                        ax_.set_ylim(0, max_value)
                plt.show()
ks = KaggleSurvey()
ks_ = KaggleSurvey()
ks.set_df_I_want_by_country_name(["If you insert the country name not in the dataset"])
ks.set_df_I_want_by_country_name(["United States of America", "India", "China"], is_need_total=True)
ks_.set_df_I_want_by_country_name(["South Korea", "Germany", "Mexico", "Italy", "Egypt", "Canada"])
ks.draw_plot(question_number=1, plot_cols=4, is_need_number_of_respondents=True)
ks_.draw_plot(question_number=1)
ks.draw_plot(question_number=9, plot_cols=2)
q9_order = [
    '0-10,000',
    '10-20,000',
    '20-30,000',
    '30-40,000',
    '40-50,000',
    '50-60,000',
    '60-70,000',
    '70-80,000',
    '80-90,000',
    '90-100,000',
    '100-125,000',
    '125-150,000',
    '150-200,000',
    '200-250,000',
    '250-300,000',
    '300-400,000',
    '400-500,000',
    '500,000+',
    'I do not wish to disclose my approximate yearly compensation'
]
ks.draw_plot(question_number=9, plot_cols=2, order=q9_order)
ks.draw_plot(question_number=18, plot_cols=2)
ks.draw_plot(question_number=18, plot_cols=2, ylim_offset=0.2)
ks.draw_plot(question_number=28, plot_cols=1, ylim_offset=0.3)
ks.draw_plot(question_number=34, plot_cols=4)
ks.draw_plot(question_number=35, plot_cols=4)
US = ks.get_df_I_want_by_country_name("United States of America")
US_MALE = US[US["Q1"] == "Male"]
US_FEMALE = US[US["Q1"] == "Female"]
dfs = {
    "US_Total" : US,
    "US_Male" : US_MALE,
    "US_Female" : US_FEMALE
}
ks_new = KaggleSurvey()
ks_new.draw_plot(35, dfs_ = dfs, ylim_offset=0.2)
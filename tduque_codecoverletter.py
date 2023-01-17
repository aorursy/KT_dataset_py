## Don't worry if you don't know any of this. Just click the blue arrow in the left and search for the output text at the end of the page.



from IPython.display import Markdown, display

def printmd(string):

    display(Markdown(string))



introduction = "to whom it may concern,\n"



printmd(introduction.upper())



def join_list_in_text(text, listed_values, append_punctuation=False):

    for num, part_text in enumerate(listed_values):

        text += part_text

        if num < len(listed_values)-2:

            text += ", "

        elif num < len(listed_values)-1:

            text += " and "

        elif append_punctuation==True:

            text += ". "

    return text



class JobPosition:

    def __init__(self, company="Dundler Mifflin", position="Python Developer"):

        '''

        The main Function, it defines the Position Company Name and Position!



        '''

        self.company = company

        self.position = position

    

    def get_dict(self):

        '''

        A getter method



        '''

        return {"company":self.company, "position":self.position}



class Person:

    def __init__(self, name):

        self.name = name

        self.qualifications = set()

        self.location = "Universe"

        

    def add_location(self, location_name):

        self.location = location_name

        

    def add_qualification(self, qualification_name):

        self.qualifications.add(qualification_name)

    

    def pretty_print_qualifications(self):

        return self._join_list_(listed_values)



    def _join_list_(self, listed_values, append_punctuation=False):

        for num, part_text in enumerate(listed_values):

            text += part_text

            if num < len(listed_values)-2:

                text += ", "

            elif num < len(listed_values)-1:

                text += " and "

            elif append_punctuation==True:

                text += ". "

        return text



## Printing code!

    

jp = JobPosition("COMPANY NAME HERE","Data Scientist/Machine Learning Engineer with NLP expertise")

        

self_presentation = "My name is {}, and it is a pleasure to provide this cover letter to {} for the position of {}.".format("Tiago Duque", jp.get_dict()['company'], jp.get_dict()['position'])



area_of_expertise = "Natural Language Processing"



self_presentation += " The position requires knowledge of {} ({}), which I've been developing for the last 3 years, since the start of my Masters Degree Research.".format(area_of_expertise, "".join([word[0] for word in area_of_expertise.split()]))



research = {"name": "A Graph based Approach for Question Answering" , "objective": "build a Question Answer System from a small corpora", "techniques": ["Knowledge Graphs", "NLP", "Natural Language Understanding"]}



self_presentation += "\n\nMy research, named \"{}\" had the aim to {}, using techniques such as ".format(research['name'],research['objective'])



self_presentation = join_list_in_text(self_presentation, research['techniques'], True)



self_presentation += "During the research, I learned to work with many tools, including "





import pandas as pd



tools_expertise_dict = {"Python": "Expert", "spaCy": "Experienced", "NLTK": "Some Knowledge", "Jupyer Notebooks": "Experienced", "Pandas":"Experienced", "OpenNLP": "Some Knowledge", "Py4j":"Some Knowledge"}

tools_df = pd.DataFrame(columns=['Tool','Expertise Level'])

tools_df['Tool'] = tools_expertise_dict.keys()

tools_df['Expertise Level'] = tools_expertise_dict.values()



self_presentation = join_list_in_text(self_presentation, tools_df['Tool'], True)



other_skills = " Aside from the Masters, I also made some courses on Machine Learning and Deep Learning, which gave me some theoretical and practical knowledge on Machine Learning Techniques. In this area I learned how to use Weka (Java), Keras and Tensorflow 2.0."



new_skills = {"Scikit Learn":"Experienced", "Keras":"Experienced", "Weka":"Experienced", "Tensorflow 2.0":"Experienced"}



self_presentation += other_skills



import spacy



stack_overflow = """\n\nI have been recently using these skills to help others at StackOverflow, since I currently work with bureaucratic activities. With that, I've managed to get third place in [NLP] tag contributions after only a pair of weeks, staying close to famous contributors and researches, such as Dr. Cristopher Manning.

\n\n I'm also making efforts in explaining NLP and its main steps in a series of Medium Articles published with aid of Analytics Vidhya, a popular Data Science enterprise in India."""



nlp = spacy.load('en')

doc = nlp(stack_overflow)



current_place = "second"



for ent in doc.ents:

    if ent.label_ == "ORG":

        stack_overflow = stack_overflow.replace(ent.text, "*"+ent.text+"*")

    elif ent.label_ == "ORDINAL":

        stack_overflow = stack_overflow.replace(ent.text, current_place)

    elif ent.label_ == "PERSON":

        stack_overflow = stack_overflow.replace(ent.text, "**"+ent.text+"**")



self_presentation+=stack_overflow



habilities = "\n\nI can currently combine several NLP pipeline activities to daily programming tasks and for the composition of Machine Learning Models, as well as propose architectural solutions for many Machine Learning and NLP problems."

habilities += " This points back to my original formation, as a software Engineer. In my graduation studies, I've learned many traits about system architecture and engineering."



new_skills.update({"Java":"Experienced", "Software Engineering/Architecture":"Some Knowledge"})



self_presentation+=habilities



languages = ["Portuguese", "English", "Spanish"]



personal_skills = "\n\nRegarding other relevant skills, I am a very communicative person. My experience as a high school teacher helped me develop my communication skills, allowing me to present complex subjects to inexperienced specialists."

personal_skills+=" I also speak {} language{} fluently: {}, which allows me to communicate in most parts of the world.".format(len(languages), 's' if len(languages)>1 else '', ", ".join(languages))



self_presentation +=personal_skills







for tool in new_skills:

    tools_df= tools_df.append({"Tool":tool, "Expertise Level":new_skills[tool]}, ignore_index=True)

    

otherCandidates_df = pd.DataFrame(columns = tools_df['Tool'])

otherCandidates_df['Fit For Position'] = 0



import random

def mockResults(columns, result_column_name='Fit', min_value = 0, max_value=10, good_values=False):

    mock_res = {}

    for column in columns:

        mock_res[column] = 0

        if column == result_column_name:

            if good_values == True:

                mock_res[column] = float(1)

            else:

                mock_res[column] = float(0)

        elif good_values == True:

            mock_res[column] = float(random.randrange(int(max_value*0.7), max_value))

        else:

            mock_res[column] = float(random.randrange(min_value, int(max_value*0.5)))

    return mock_res



import warnings

warnings.filterwarnings('ignore')



total_candidates = 100

for i in range(total_candidates):

    otherCandidates_df=otherCandidates_df.append(mockResults(otherCandidates_df.columns, 'Fit For Position', good_values=i > total_candidates/2), ignore_index=True)



from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

import numpy as np



X = np.array(otherCandidates_df[otherCandidates_df.columns[:-1]])

y = np.array(otherCandidates_df[otherCandidates_df.columns[-1]])

rfc = RandomForestClassifier(n_estimators=10)

svc = SVC(kernel='linear')

knn = KNeighborsClassifier(n_neighbors=5)

nb = GaussianNB()

lr = LogisticRegression()



ensemble = VotingClassifier(estimators=[("Random forest", rfc), ("KNN",knn), ("Naive Bayes", nb), ("SVC",svc), ("Logistic Reg.",lr)])

ensemble.fit(X,y)



def category_as_value(category):

    if category.lower() == "expert":

        return 10

    elif category.lower() == "experienced":

        return 7

    elif category.lower() == "some knowledge":

        return 5

    else:

        return 0



tools_df['Values'] = tools_df['Expertise Level'].apply(lambda x: category_as_value(x))



is_fit = ensemble.predict([tools_df['Values']])



if bool(is_fit[0]):

    self_presentation += "\n\n With all considered, I find myself fit for the position, being able to add value to the company and accomplish the required tasks for the position. Below is a table summarizing my skills and expertise levels."

    self_presentation += " Also follows the links for the mentioned article series in Medium, my StackOverflow Profile and my Github profile. Best Regards, \n\n Tiago Duque"

    links = [("Medium head article","https://medium.com/@tfduque/dissecting-natural-language-processing-layer-by-layer-an-introductory-overview-d11cfff4f329"),("StackOverflow Profile","https://stackoverflow.com/users/3288004/tiago-duque?tab=profile"),

            ("GitHub Profile", "https://github.com/Sirsirious"),("Linkedin Profile","https://www.linkedin.com/in/tfduque/?locale=en_US")]

    for page, link in links:

        self_presentation += "\n\n{}: {}".format(page, link)

        

printmd(self_presentation)



from IPython.display import display



display(tools_df[tools_df.columns[:-1]])

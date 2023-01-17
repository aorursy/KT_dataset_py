import pandas as pd

from IPython.core.display import display, HTML



# Import data

questions_only_19 = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv")



# Iterate questions and add the index and value formatted in HTML to a string variable

# The questions table is transposed and the first row discarted

questionnaire = ""

for index, value in questions_only_19.T[1:].iterrows():

    questionnaire += f"<strong>{index}</strong>: {value[0]}<br>\n"



# Save the content in an HTML file to have it in output

with open("questionnaire.html", "w") as file: file.write(questionnaire)



# Display the HTML content

display(HTML(questionnaire))
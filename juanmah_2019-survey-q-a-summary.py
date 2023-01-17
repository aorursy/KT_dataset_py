# !pip install nb_black

# %load_ext nb_black

import pandas as pd

from IPython.core.display import display, HTML



# Import data

data_path = "../input/kaggle-survey-2019/"

multiple_choice_responses_19 = pd.read_csv(

    data_path + "multiple_choice_responses.csv", engine="c", low_memory=False

)

other_text_responses_19 = pd.read_csv(

    data_path + "other_text_responses.csv", engine="c"

)

questions_only_19 = pd.read_csv(data_path + "questions_only.csv", engine="c")
# TABLE OF CONTENTS



# Iterate questions and add links

# The questions table is transposed and the first row discarted

questionnaire = "<h1>Table of contents</h1><br>"

for index, value in questions_only_19.T[1:].iterrows():

    questionnaire += f'<a href="#{index}" style="text-decoration: none;"><strong>{index}</strong>: {value[0]}</a><br>\n'

# Display the HTML content

display(HTML(questionnaire))
# Iterate questions and add tables formatted in HTML to a string variable

# The questions table is transposed and the first row discarted

questionnaire = ""

for index, value in questions_only_19.T[1:].iterrows():



    # QUESTION

    # --------

    questionnaire += (

        f'<span id="{index}"><strong>{index}</strong></span>: {value[0]}<br>\n'

    )



    # NORMAL QUESTION

    # ---------------

    # If the question is single choice, it is in the multiple_choice_responses columns

    if index in multiple_choice_responses_19.columns:

        # Get the responses sorted by the frequency in descending order

        question = multiple_choice_responses_19[index][1:].value_counts()

        # Check if they are more than 10 responses

        # Get the sum of the rest

        more_sum = question[10:].sum()

        # Get the number of discarded responses

        more_number = len(question[10:])

        # If they are more than 11 responses:

        # - show only the 10 first responses

        # - aggregate the rest and show the summed frequency

        if more_number > 1:

            question = question[0:10].append(

                pd.Series([more_sum], index=[f"{more_number} more ..."])

            )

        # Add the table formatted in HTML to the summary

        questionnaire += question.to_frame().to_html(header=False, bold_rows=False)

        questionnaire += "<br>\n"



    # MULTIPLE CHOICE QUESTION

    # ------------------------

    # If the question is multiple choice, the question is followed by _Part_

    # Get all columns with _Part_

    multiple_choice = multiple_choice_responses_19.filter(

        like=f"{index}_Part_", axis="columns"

    )

    # Drop multiple choice columns with text response

    multiple_choice = multiple_choice.drop(

        columns=list(multiple_choice.filter(like="_TEXT"))

    )

    # If it is an adequate multiple choice question, print table

    # In instance, Q14 only have parts with free text responses

    if multiple_choice.shape[1] > 0:

        questionnaire += (

            multiple_choice[1:]  # discard first row

            .describe()  # describe do a summary with top and count

            .T[["top", "count"]]  # transpose and get only top and count

            .sort_values(by="count", ascending=False)  # sorty by count

            .to_html(index=False, header=False)  # export to HTML

        )

        questionnaire += "<br>\n"



    # TEXT RESPONSES

    # --------------

    # If the question has a free text, the question ends with _TEXT

    # Get all columns ending with _TEXT

    other = other_text_responses_19.filter(regex=f"{index}_.*_TEXT", axis="columns")

    # Make a table of each free text field

    for o in other.columns:

        # Get the responses sorted by the frequency in descending order

        unique = other[o].value_counts()

        # Check if they are more than 10 responses

        # Get the sum of the rest

        more_sum = unique[10:].sum()

        # Get the number of discarded responses

        more_number = len(unique[10:])

        # If they are more than 11 responses:

        # - show only the 10 first responses

        # - aggregate the rest and show the summed frequency

        if more_number > 1:

            unique = unique[0:10].append(

                pd.Series([more_sum], index=[f"{more_number} more ..."])

            )

        # Add the table formatted in HTML to the summary

        questionnaire += unique.to_frame(name=o).to_html(header=True, bold_rows=False)

        questionnaire += "<br>\n"



# Save the content in an HTML file to have it in output

with open("questionnaire.html", "w") as file:

    file.write(questionnaire)



# Display the HTML content

display(HTML(questionnaire))
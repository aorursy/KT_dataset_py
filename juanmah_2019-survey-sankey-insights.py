# !pip install nb_black

# %load_ext nb_black

import pandas as pd

import numpy as np

from IPython.core.display import display, HTML

import plotly.graph_objects as go



# Import data

data_path = "../input/kaggle-survey-2019/"

multiple_choice_responses_19 = pd.read_csv(

    data_path + "multiple_choice_responses.csv", engine="c", low_memory=False

)

questions_only_19 = pd.read_csv(data_path + "questions_only.csv", engine="c")
def plot_sankey(questions, threshold=0, diff=False):

    """This function plots a Sankey diagram from a list of two or more questions.



    Parameters

    ----------

    questions: list of str

        List of the questions to be plotted.

    threshold: int

        Relations below this value will be discarded. It's useful to simplify the diagram.



    Returns

    -------

    None



    """



    def get_responses_and_series(question):

        """This function gets all the responses for a question and a Series por each response.



        Notes

        -----

        They are single choice questions that has all the responses in a single Series.

        This Series is broke down in a Series of each response.



        The are multiple choice questions that has one Series per response.



        Parameters

        ----------

        question: str

            Question to be processed.



        Returns

        -------

        responses: list of str

            It is a list of all responses for that question.

        series: DataFrame

            It is a DataFrame with a Series por each response.

        """

        # Initialize series DataFrame

        series = pd.DataFrame()



        # NORMAL QUESTION

        # ---------------

        # If the question is single choice, it is in the multiple_choice_responses columns

        if question in multiple_choice_responses_19.columns:

            # Get the list of responses sorted by the frequency in descending order

            responses = (

                multiple_choice_responses_19[question][1:].value_counts().index.tolist()

            )

            # Iterate the responses, breaking down the series for each response

            for response in responses:

                # Make a copy

                response_series = multiple_choice_responses_19[question][1:].copy()

                # Make null other responses

                response_series[response_series != response] = np.NaN

                # Add this response series to the series DataFrame

                series = pd.concat(

                    [series, response_series], axis="columns", sort=False

                )



        # MULTIPLE CHOICE QUESTION

        # ------------------------

        # If the question is multiple choice, the question is followed by _Part_

        # Get all columns with _Part_ and discard free text columns which have _TEXT

        else:

            # Check if the question is multiple choice

            multiple_choice = multiple_choice_responses_19[1:].filter(

                like=f"{question}_Part_", axis="columns"

            )

            # Don't use multiple choice with text response

            multiple_choice = multiple_choice.drop(

                columns=list(multiple_choice.filter(like="_TEXT"))

            )

            # Get the list of responses sorted by the frequency in descending order

            responses = multiple_choice.describe().T["top"].tolist()

            #

            series = multiple_choice



        # Rename the series names

        series.columns = responses



        return responses, series



    def diff_color(diff):

        """This function returns a greenish color if diff is greater than 0 

        and a redish color if diff is lesser than 0.

        

        Notes

        -----

        First, the diff is scaled and transformed from (-inf…+inf) to (-1…1).

        Then, (-1…1) is transformed to an hex color, where 0 is light grey, 1 green and -1 red.



        Parameters

        ----------

        diff: float

            Difference in parts per unit of the number of real links compared with theorical neutral links.

            0.1 means that real links are a 10% greater than an equilibrated weight

            -0.1 means that real links are a 10% lesser than an equilibrated weight



        Returns

        -------

        color: str

            Color in hex format. In instance: "#ff0000"

        """

        diff = diff / 4

        diff = diff / (1 + abs(diff))

        if diff >= 0:

            color = f"#{int((1-diff)*192 + diff*0):02x}{int((1-diff)*192 + diff*255):02x}{int((1-diff)*192 + diff*0):02x}"

        else:

            diff = -diff

            color = f"#{int((1-diff)*192 + diff*255):02x}{int((1-diff)*192 + diff*0):02x}{int((1-diff)*192 + diff*0):02x}"

        return color



    # Initialize variables

    questions_index = 0

    source_offset = 0

    link = pd.DataFrame(columns={"source", "target", "value"})



    # Get the question of the source side

    source_question = questions[questions_index]

    # Get the responses and series of the source side

    source_responses, source_series = get_responses_and_series(source_question)

    # Set the first part of the title

    title = f"\n<br><b>{source_question}</b>: {questions_only_19[source_question].values[0]}"

    # Fill the node list with the nodes of the source side

    node = source_responses.copy()



    # For the remaining questions

    while questions_index < (len(questions) - 1):

        # Increment the questions index

        questions_index += 1

        # Initialize variables

        link_partial = pd.DataFrame(columns={"source", "target", "value"})

        # Get the question of the target side

        target_question = questions[questions_index]

        # Get the responses and series of the target side

        target_responses, target_series = get_responses_and_series(target_question)

        # Append the part of the title relative to the current question

        title += f"\n<br><b>{target_question}</b>: {questions_only_19[target_question].values[0]}"



        # The target nodes will begin in the next position of the current node list

        target_offset = len(node)

        # The node list is extended with the target responses list

        node += target_responses



        # Iterate all the combinations between the source and target sides

        for source_index, source_response in enumerate(source_responses):

            for target_index, target_response in enumerate(target_responses):

                # Value is the number of coincidences of the source/target combination.

                # A coincidence occurs when for the given combination and the same respondent

                # there are responses in both sides.

                value = (

                    source_series[source_response].notna()

                    & target_series[target_response].notna()

                ).sum()

                # Only relations with values greater than a threshold are processed

                if value >= threshold:

                    # Append the link DataFrame with the source, target and value fields.

                    link_partial = link_partial.append(

                        {

                            "source": source_offset + source_index,

                            "target": target_offset + target_index,

                            "value": value,

                        },

                        ignore_index=True,

                    )

        # If diff is colored

        if diff:

            # Get the source sum of links for each node

            source_size = (

                link_partial[["source", "value"]]

                .groupby("source")

                .agg(np.sum)

                .rename(columns={"value": "count"})

            )



            # Get the source sum of links for each node

            target_size = (

                link_partial[["target", "value"]]

                .groupby("target")

                .agg(np.sum)

                .rename(columns={"value": "count"})

            )



            # Get the total links from source or target nodes (they are the same)

            link_size = source_size.sum().values[0]



            # Calculate the weights of links of a neutral and equilibrated net

            base = (

                source_size.dot(target_size.T)

                .div(link_size)

                .reset_index()

                .melt(id_vars="source")

            )



            # As the link_partial and base are not indexed with the same indices,

            # the coupling has to be done by hand.

            link_partial["base"] = link_partial.apply(

                lambda x: base[

                    (base["target"] == x["target"]) & (base["source"] == x["source"])

                ]["value"].values[0],

                axis="columns",

            )



            # Calculate the diff

            link_partial["diff"] = (

                link_partial["value"] - link_partial["base"]

            ) / link_partial["base"]



            # Get the color

            link_partial["color"] = link_partial["diff"].apply(diff_color)



        # Append the partial link to the general one

        link = pd.concat([link, link_partial], sort=False)



        # Prepare the target side to be the next source side

        source_responses = target_responses

        source_series = target_series

        source_offset = target_offset



    # If diff is colored

    if diff:

        color = link["color"]

    else:

        color = None

    # Create and show the diagram

    fig = go.Figure(

        data=[

            go.Sankey(

                node=dict(label=node),

                link=dict(

                    source=link["source"],

                    target=link["target"],

                    value=link["value"],

                    color=color,

                ),

            )

        ]

    )

    fig.update_layout(title_text=f"Sankey Diagram: {title}", font_size=10, height=800)

    fig.show()
plot_sankey(["Q18", "Q19"], threshold=100, diff=True)
plot_sankey(["Q14", "Q19"], threshold=40, diff=True)
plot_sankey(["Q14", "Q19"], threshold=40, diff=True)
plot_sankey(["Q14", "Q20"], threshold=200, diff=True)
plot_sankey(["Q18", "Q20"], threshold=1000, diff=True)
plot_sankey(["Q15", "Q23"], threshold=40)
plot_sankey(["Q31", "Q29", "Q30"], threshold=150)
plot_sankey(["Q21", "Q22"], threshold=0, diff=True)
plot_sankey(["Q21", "Q24"], threshold=1000, diff=True)
plot_sankey(["Q12", "Q13"], threshold=1000, diff=True)
plot_sankey(["Q13", "Q14"], threshold=200, diff=True)
plot_sankey(["Q10", "Q11"], threshold=0, diff=True)
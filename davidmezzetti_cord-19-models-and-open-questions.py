from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import run



task = """

id: 4

name: models_and_open_questions



# Field definitions

fields:

    common: &common

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type



    methods: &methods

        - {name: Method, query: confirmation method, question: What rna confirmation method used}

        - {name: Result, query: $QUERY conclusions findings results, question: What is conclusion on $QUERY}

        - {name: Measure of Evidence, constant: "-"}



    appendix: &appendix

        - name: Sample Size

        - name: Sample Text

        - name: Study Population

        - name: Matches

        - name: Entry



Are there studies about phenotypic change_:

    query: Phenotypic genetic change

    columns:

        - *common

        - *methods

        - *appendix



Efforts to develop qualitative assessment frameworks:

    query: qualitative assessment frameworks

    columns:

        - *common

        - {name: Addressed Population, query: $QUERY, question: What group studied}

        - {name: Challenge, query: $QUERY, question: What challenge discussed}

        - {name: Solution, query: solutions recommendations interventions, question: What is solution} 

        - {name: Measure of Evidence, constant: "-"} 

        - *appendix



How can we measure changes in COVID-19_s behavior in a human host as the virus evolves over time_:

    query: Human immune response to COVID-19

    columns:

        - *common

        - *methods

        - *appendix



Serial Interval (time between symptom onset in infector-infectee pair):

    query: Serial Interval (for infector-infectee pair)

    columns:

        - *common

        - {name: Age, query: median patient age, question: What is median patient age}

        - {name: Sample Obtained, query: throat respiratory fecal sample, question: What sample}

        - {name: Serial Interval (days), query: serial interval days, question: What is median serial interval}

        - *appendix



Studies to monitor potential adaptations:

    query: studies monitor adaptations

    columns:

        - *common

        - *methods

        - *appendix



What do models for transmission predict_:

    query: Transmission model predictions

    columns:

        - *common

        - {name: Method, query: statistical model approach, question: What model used}

        - {name: Excerpt, query: r0 predict, question: What is model prediction}

        - *appendix



What is known about adaptations (mutations) of the virus_:

    query: Virus mutations

    columns:

        - *common

        - *methods

        - *appendix



What regional genetic variations (mutations) exist:

    query: Regional genetic variations (mutations)

    columns:

        - *common

        - *methods

        - *appendix

"""



# Build and display the report

run(task)
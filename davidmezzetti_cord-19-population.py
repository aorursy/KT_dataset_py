from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import run



task = """

id: 1

name: population



# Field definitions

fields:

    common: &common

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type



    population: &population

        - {name: Addressed Population, query: $QUERY, question: What group studied}

        - {name: Challenge, query: $QUERY, question: What challenge discussed}

        - {name: Solution, query: solutions recommendations interventions, question: What is solution} 

        - {name: Measure of Evidence, constant: "-"} 



    appendix: &appendix

        - name: Sample Size

        - name: Sample Text

        - name: Study Population

        - name: Matches

        - name: Entry



    columns: &columns

        - *common

        - *population

        - *appendix



# Define query tasks

Management of patients who are underhoused or otherwise lower social economic status:

    query: patients poor, homeless, lower social economic status

    columns: *columns



Measures to reach marginalized and disadvantaged populations:

    query: Communicating with marginalized disadvantaged populations

    columns: *columns



Methods to control the spread in communities:

    query: prevent spread in communities

    columns: *columns



Modes of communicating with target high-risk populations:

    query: Communicating with target high-risk populations - elderly, health care workers

    columns: *columns



What are recommendations for combating_overcoming resource failures_:

    query: mitigate resource shortages

    columns: *columns



What are ways to create hospital infrastructure to prevent nosocomial outbreaks_:

    query: prevent nosocomial outbreaks in hospitals

    columns: *columns

"""



# Build and display the report

run(task)
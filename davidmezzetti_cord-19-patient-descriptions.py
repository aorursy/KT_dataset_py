from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import run



task = """

id: 3

name: patient_descriptions



# Field definitions

fields:

    common: &common

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type



    asymptomatic: &asymptomatic

        - {name: Age, query: median patient age, question: What is median patient age}

        - {name: Sample Obtained, query: throat respiratory fecal sample, question: What sample}

        - {name: Asymptomatic Transmission, query: proportion percent asymptomatic patients, question: What percent asymptomatic}

        - {name: Excerpt, query: proportion percent asymptomatic patients, question: What percent asymptomatic, snippet: true}



    incubation: &incubation

        - {name: Age, query: median patient age, question: What is median patient age}

        - {name: Days, query: range of incubation period days, question: What is median incubation period}

        - {name: Range (Days), query: range of incubation periods days, question: What is incubation period range}



    appendix: &appendix

        - name: Sample Size

        - name: Sample Text

        - name: Study Population

        - name: Matches

        - name: Entry



Can the virus be transmitted asymptomatically or during the incubation period_:

    query: Asymptomatic transmission

    columns:

        - *common

        - *asymptomatic

        - *appendix



How does viral load relate to disease presentations and likelihood of a positive diagnostic test_:

    query: viral load relation to positive diagnostic test

    columns:

        - *common

        - {name: Age, query: median patient age, question: What is median patient age}

        - {name: Sample Obtained, query: throat respiratory fecal sample, question: What sample}

        - {name: Excerpt, query: $QUERY, question: What is $QUERY, snippet: True}

        - *appendix



Incubation period across different age groups:

    query: Incubation period children adult elderly days

    columns:

        - *common

        - *incubation

        - *appendix



Length of viral shedding after illness onset:

    query: Shedding duration days

    columns:

        - *common

        - *incubation

        - *appendix



Manifestations of COVID-19 including but not limited to possible cardiomyopathy and cardiac arrest:

    query: virus related manifestations

    columns:

        - *common

        - {name: Age, query: median patient age, question: What is median patient age}

        - {name: Sample Obtained, query: throat respiratory fecal sample, question: What sample}

        - {name: Manifestation, query: clinical manifestations, question: What manifestations}

        - {name: Frequency of Symptoms, query: symptoms frequency, question: What was frequency of symptoms}

        - {name: Excerpt, query: clinical manifestations, question: What manifestations, snippet: True}

        - *appendix



Proportion of all positive COVID19 patients who were asymptomatic:

    query: Proportion of asymptomatic patients

    columns:

        - *common

        - *asymptomatic

        - *appendix



Proportion of pediatric COVID19 patients who were asymptomatic:

    query: Asymptomatic pediatric patients

    columns:

        - *common

        - *asymptomatic

        - *appendix



What is the incubation period of the virus_:

    query: Range of incubation periods days

    columns:

        - *common

        - *incubation

        - *appendix

"""



# Build and display the report

run(task)

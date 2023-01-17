from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import run



task = """

id: 2

name: relevant_factors



# Field definitions

fields:

    common: &common

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type



    containment: &containment

        - {name: Factors, query: $QUERY, question: what containment method}

        - {name: Influential, constant: "-"}

        - {name: Excerpt, query: $QUERY, question: what containment method, snippet: true}

        - {name: Measure of Evidence, query: countries cities, question: What locations}



    weather: &weather

        - {name: Factors, query: temperature humidity, question: What weather factor}

        - {name: Influential, constant: "-"}

        - {name: Excerpt, query: temperature humidity, question: How weather effects virus}

        - {name: Measure of Evidence, query: countries cities, question: What locations}



    appendix: &appendix

        - name: Sample Size

        - name: Sample Text

        - name: Study Population

        - name: Matches

        - name: Entry



    columns: &columns

        - *common

        - *containment

        - *appendix



# Define query tasks

Effectiveness of a multifactorial strategy to prevent secondary transmission: 

    query: Multifactorial strategy prevent transmission effect

    columns: *columns



Effectiveness of case isolation_isolation of exposed individuals to prevent secondary transmission:

    query: Case isolation exposed individuals, quarantine effect

    columns: *columns



Effectiveness of community contact reduction:

    query: Community contact reduction effect

    columns: *columns



Effectiveness of inter_inner travel restriction:

    query: Travel restrictions effect

    columns: *columns



Effectiveness of school distancing:

    query: School distancing effect

    columns: *columns



Effectiveness of workplace distancing to prevent secondary transmission:

    query: Workplace distancing effect

    columns: *columns



Evidence that domesticated_farm animals can be infected and maintain transmissibility of the disease:

    query: Evidence that domesticated, farm animals can be infected and maintain transmissibility of the disease

    columns:

        - *common

        - {name: Factors, query: animals studied, question: what animals}

        - {name: Influential, constant: "-"}

        - {name: Excerpt, query: animals studied, question: "Can animals transmit SARS-COV-2"}

        - {name: Measure of Evidence, query: confirmation method, question: What rna confirmation method used}

        - *appendix



How does temperature and humidity affect the transmission of 2019-nCoV_:

    query: Temperature, humidity environment affect on transmission

    columns:

        - *common

        - *weather

        - *appendix



Methods to understand and regulate the spread in communities:

    query: Methods to regulate the spread in communities

    columns: *columns



Seasonality of transmission:

    query: Seasonality of transmission significant factors and effect

    columns:

        - *common

        - *weather

        - *appendix



What is the likelihood of significant changes in transmissibility in changing seasons_:

    query: transmission changes with seasonal change

    columns:

        - *common

        - *weather

        - *appendix

"""



# Build and display the report

run(task)
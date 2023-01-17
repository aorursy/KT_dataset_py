from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import run



task = """

id: 7

name: therapeutics_interventions_and_clinical_studies



# Field definitions

fields:

    common: &common

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type



    therapeutics: &therapeutics

        - {name: Therapeutic method(s) utilized/assessed, query: therapeutic method, question: What therapeutic method}

        - {name: Severity of Disease, query: patient severity icu, question: What is patient severity}

        - {name: General Outcome/Conclusion Excerpt, query: therapeutic method outcome, question: What is conclusion on therapeutic method, snippet: True}

        - {name: Primary Endpoint(s) of Study, query: patient outcome, question: What is patient outcome, snippet: True}



    appendix: &appendix

        - name: Sample Size

        - name: Sample Text

        - name: Study Population

        - name: Matches

        - name: Entry



    columns: &columns

        - *common

        - *therapeutics

        - *appendix



What is the best method to combat the hypercoagulable state seen in COVID-19_:

    query: therapeutic method hypercoagulable

    columns: *columns



What is the efficacy of novel therapeutics being tested currently_:

    query: novel therapeutics efficacy

    columns: *columns

"""



# Build and display the report

run(task)
from cord19reports import install



# Install report dependencies

install()

%%capture --no-display

from cord19reports import run



task = """

id: 6

name: diagnostics



# Field definitions

fields:

    common: &common

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type



    diagnostics: &diagnostics

        - {name: Detection Method, query: confirmation method, question: What rna confirmation method used}

        - {name: Measure of Testing Accuracy, query: sensitivity specificity accuracy, question: What is assay detection accuracy}

        - {name: Speed of Assay, query: turnaround time minute hour, question: What is assay detection speed}



    appendix: &appendix

        - name: Sample Size

        - name: Sample Text

        - name: Study Population

        - name: Matches

        - name: Entry



    columns: &columns

        - *common

        - *diagnostics

        - *appendix



# Define queries

Development of a point-of-care test and rapid bed-side tests:

    query: Development of a point-of-care test and rapid bed-side detection methods

    columns: *columns



Diagnosing SARS-COV-2 with antibodies:

    query: diagnose sars-cov-2 antibodies

    columns: *columns



Diagnosing SARS-COV-2 with Nucleic-acid based tech:

    query: diagnose sars-cov-2 nucleic-acid

    columns: *columns

"""



# Build and display the report

run(task)
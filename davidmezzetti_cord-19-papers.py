%%capture

from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import report, render



task = """

name: query



cord-19:

    query: +cord-19

    columns:

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type

        - {name: Analysis, query: model ai learning nlp, question: What methods used to analyze data}

        - name: Sample Size

        - name: Study Population

        - name: Matches

        - name: Entry

"""



# Build and render report

report(task)

render("query")
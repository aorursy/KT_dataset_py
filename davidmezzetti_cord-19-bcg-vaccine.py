%%capture

from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import report, render



task = """

name: query



bcg vaccine:

    query: +bcg vaccine trial

    columns:

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type

        - {name: BGC Vaccine Effective, query: bcg vaccine covid-19, question: Does BCG vaccine protect against COVID-19, snippet: True}

        - name: Sample Size

        - name: Study Population

        - name: Matches

        - name: Entry

"""



# Build and render report

report(task)

render("query")
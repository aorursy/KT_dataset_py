%%capture

from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import report, render



task = """

name: query



antiviral covid-19 success treatment:

    query: antiviral covid-19 success treatment

    columns:

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type

        - {name: Country, query: what country}

        - {name: Drugs, query: what drugs tested}

        - name: Sample Size

        - name: Study Population

        - name: Matches

        - name: Entry

"""



# Build and render report

report(task)

render("query")
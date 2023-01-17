%%capture

from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import report, render



task = """

name: forecasting



Forecasting and Modeling:

    query: forecasting and modeling

    columns:

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type

        - name: Sample Size

        - name: Study Population

        - name: Matches

        - name: Entry

"""



# Build and render report

report(task)

render("forecasting")
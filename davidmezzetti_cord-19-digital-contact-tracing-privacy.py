%%capture

from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import report, render



task = """

name: query



digital contact tracing privacy:

    query: digital mobile contact tracing +privacy

    columns:

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type

        - {name: Tracing Method, query: wifi network bluetooth blockchain beacon os app, question: What technical method}

        - {name: Privacy Issues, query: privacy concerns, question: What privacy concerns}

        - {name: Privacy Protections, query: privacy protection mitigations, question: What methods to preserve privacy, snippet: True}

        - name: Sample Size

        - name: Study Population

        - name: Matches

        - name: Entry

"""



# Build and render report

report(task)

render("query")
from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import run



task = """

id: 8

name: risk_factors



# Field definitions

fields:

    common: &common

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type



    severity: &severity

        - {name: Severe, query: $QUERY, question: What is $NAME risk number}

        - {name: Severe lower bound, query: $QUERY, question: What is $NAME range minimum}

        - {name: Severe upper bound, query: $QUERY, question: What is $NAME range maximum}

        - {name: Severe p-value, query: $QUERY, question: What is the $NAME p-value}



    appendix: &appendix

        - name: Sample Size

        - name: Sample Text

        - name: Study Population

        - name: Matches

        - name: Entry



    columns: &columns

        - *common

        - *severity

        - *appendix



Age:

    query: +age ci

    columns: *columns



Asthma:

    query: +asthma ci

    columns: *columns



Autoimmune disorders:

    query: +autoimmune disorders ci

    columns: *columns



Cancer:

    query: +cancer ci

    columns: *columns



Cardio- and cerebrovascular disease:

    query: cardio and +cerebrovascular disease ci

    columns: *columns



Cerebrovascular disease:

    query: +cerebrovascular disease ci

    columns: *columns



Chronic digestive disorders:

    query: +digestive disorders ci

    columns: *columns



Chronic kidney disease:

    query: +kidney disease ckd ci

    columns: *columns



Chronic liver disease:

    query: +liver disease ci

    columns: *columns



Chronic respiratory diseases:

    query: chronic +respiratory disease ci

    columns: *columns



COPD:

    query: chronic obstructive pulmonary disease +copd ci

    columns: *columns



Dementia:

    query: +dementia ci

    columns: *columns



Diabetes:

    query: +diabetes ci

    columns: *columns



Drinking:

    query: +alcohol abuse ci

    columns: *columns



Endocrine diseases:

    query: +endocrine disease ci

    columns: *columns



Ethnicity_ Hispanic vs. non-Hispanic:

    query: +hispanic race ci

    columns: *columns



Heart Disease:

    query: +heart +disease ci

    columns: *columns



Heart Failure:

    query: +heart +failure ci

    columns: *columns



Hypertension:

    query: +hypertension ci

    columns: *columns



Immune system disorders:

    query: +immune system disorder ci

    columns: *columns



Male gender:

    query: +male ci

    columns: *columns



Neurological disorders:

    query: +neurological disorders ci

    columns: *columns



Overweight or obese:

    query: +overweight obese ci

    columns: *columns



Race_ Asian vs. White:

    query: race +asian +white ci

    columns: *columns



Race_ Black vs. White:

    query: race +black +white ci

    columns: *columns



Race_ Other vs. White:

    query: race +white ci

    columns: *columns



Respiratory system diseases:

    query: +respiratory disease ci

    columns: *columns



Smoking Status:

    query: +smoking smoker ci

    columns: *columns

"""



# Build and display the report

run(task)
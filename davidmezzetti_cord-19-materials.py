from cord19reports import install



# Install report dependencies

install()
%%capture --no-display

from cord19reports import run



task = """

id: 5

name: materials



# Field definitions

fields:

    common: &common

        - name: Date

        - name: Study

        - name: Study Link

        - name: Journal

        - name: Study Type



    appendix: &appendix

        - name: Sample Size

        - name: Sample Text

        - name: Study Population

        - name: Matches

        - name: Entry



# Define queries

Adhesion to hydrophilic_phobic surfaces:

    query: Persistence hydrophilic, phobic plastic, steel, copper, cardboard

    columns:

        - *common

        - {name: Material, query: surfaces materials tested, question: What materials tested}

        - {name: Method, query: rt-pcr confirmed, question: what rna confirmation method used}

        - {name: Virus Titer at End of Experiment, query: virus tcid tcid50, question: What viral load measured}

        - {name: Surface Type, query: hydrophobic hydrophilic, question: Is surface hydrophobic or hydrophilic}

        - {name: Conclusion, query: surfaces conclusions findings results, question: What is conclusion on surfaces}

        - {name: Measure of Evidence, query: modeling, question: What model used}

        - *appendix



Coronavirus susceptibility to heat light and radiation:

    query: coronavirus heat light radiation

    columns:

        - *common

        - {name: Material, query: heat light radiation disinfectants proposed tested, question: What heat/light/radiation/disinfectant used}

        - {name: Method, constant: "-"}

        - {name: Effective Against, query: objects materials surfaces disinfected, question: What objects disinfected}

        - {name: Effective Concentration/dosage, query: concentration dosage amount, question: What is effective concentration}

        - {name: Conclusion, query: heat light radiation disinfectants findings results, question: What is conclusion on heat/light/radiation/disinfectant method}

        - {name: Measure of Evidence, constant: "-"}

        - *appendix



How long can other HCoV strains remain viable on common surfaces_:

    query: Duration viable on surfaces days

    columns:

        - *common

        - {name: Material, query: surfaces materials tested, question: What materials tested}

        - {name: Method, constant: "-"}

        - {name: Virus Titer, query: virus tcid tcid50, question: What viral load measured}

        - {name: Persistence, query: persistence time on surface, question: What is persistence time on surface}

        - {name: Conclusion, query: surfaces conclusions findings results, question: What is conclusion on surfaces}

        - *appendix



Persistence of virus on surfaces of different materials:

    query: Persistence on surfaces days

    columns:

        - *common

        - {name: Material, query: surfaces materials tested, question: What materials tested}

        - {name: Method, query: modeling, question: What model used}

        - {name: TCID50, query: virus tcid tcid50, question: What viral load measured}

        - {name: Persistence, query: persistence time on surface, question: What is persistence time on surface}

        - {name: Conclusion, query: surfaces conclusions findings results, question: What is conclusion on surfaces}

        - *appendix



Susceptibility to environmental cleaning agents:

    query: Concentration ppm % disinfectant agent inactivate

    columns:

        - *common

        - {name: Material, query: heat light radiation disinfectants proposed tested, question: What heat/light/radiation/disinfectant used}

        - {name: Method, query: rt-pcr confirmed, question: what rna confirmation method used}

        - {name: Intended use, query: surfaces materials tested, question: What materials tested}

        - {name: Concentration/Dose, query: concentration dosage amount, question: What is effective concentration}

        - {name: Exposure time, constant: "-"}

        - {name: Conclusion, query: heat light radiation disinfectants findings results, question: What is conclusion on heat/light/radiation/disinfectant method}

        - {name: Measure of Evidence, query: rt-pcr confirmed, question: what rna confirmation method used}

        - *appendix



What do we know about viral shedding in blood_:

    query: Shedding blood days

    columns:

        - *common

        - {name: Material, constant: Blood}

        - {name: Method, query: rt-pcr confirmed, question: what rna confirmation method used}

        - {name: Days After Onset/Admission (+) Covid-19 Presence (maximum unless otherwise stated), query: shedding blood days, question: what is shedding time in blood}

        - {name: Conclusion, query: stool shedding conclusions findings results, question: What is conclusion on blood shedding}

        - *appendix



What do we know about viral shedding in stool_:

    query: Shedding stool days

    columns:

        - *common

        - {name: Material, constant: Fecal Matter}

        - {name: Method, query: rt-pcr confirmed, question: what rna confirmation method used}

        - {name: Days After Onset/Admission (+) Covid-19 Presence (maximum unless otherwise stated), query: shedding stool days, question: what is shedding time in stool}

        - {name: Conclusion, query: stool shedding conclusions findings results, question: What is conclusion on stool shedding}

        - *appendix



What do we know about viral shedding in the nasopharynx_:

    query: Shedding nasopharynx days

    columns:

        - *common

        - {name: Material, constant: Nasopharynx}

        - {name: Method, query: rt-pcr confirmed, question: what rna confirmation method used}

        - {name: Days After Onset/Admission (+) Covid-19 Presence (maximum unless otherwise stated), query: shedding nasopharynx days, question: what is shedding time in nasopharynx}

        - {name: Conclusion, query: stool shedding conclusions findings results, question: What is conclusion on nasopharynx shedding}

        - *appendix



What do we know about viral shedding in urine_:

    query: Shedding urine days

    columns:

        - *common

        - {name: Material, constant: Urine}

        - {name: Method, query: rt-pcr confirmed, question: what rna confirmation method used}

        - {name: Days After Onset/Admission (+) Covid-19 Presence (maximum unless otherwise stated), query: shedding urine days, question: what is shedding time in urine}

        - {name: Conclusion, query: stool shedding conclusions findings results, question: What is conclusion on urine shedding}

        - *appendix

"""



# Build and display the report

run(task)
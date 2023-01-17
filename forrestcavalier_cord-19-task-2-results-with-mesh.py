# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from IPython.core.display import display, HTML, Javascript

from string import Template

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import json, random

import IPython.display



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


htmlTaskDetails="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>



<h2>CORD-19 Task Details</h2>

Source: <a href="https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=558">What do we know about COVID-19 risk factors?</A>



<p><strong>What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?</strong></p>

<p>Specifically, we want to know what the literature reports about:</p>





<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>

<table class=l border=1><tr><th>Kaggle prompt<th>Search terms used<th>Formatted Results

<tr><td>Data on potential risks factors: Smoking, pre-existing pulmonary disease<td>smoking OR risk OR COPD OR asthma<td>Task2a results below

<tr><td>Data on potential risks factors: Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities<td>co-infection OR comorbidity  OR comorbidities OR  co-morbidities OR co-morbidity<td>Task2b results below

<tr><td>Data on potential risks factors: Neonates and pregnant women<td>pregnancy OR pregnant OR neonate OR newborn OR gestation OR fetus<td>Task2c results below

<tr><td>Data on potential risks factors: Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.<td>economic OR socio-economic OR socioeconomic OR behavioral OR poverty<td>Task2d results below

<tr><td>Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors<td>basic reproductive number OR  incubation OR serial OR transmission OR environment OR environmental OR environs OR surroundings<td>Task2e results below

<tr><td>Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups<td>severity OR fatality AND hospitalized OR high-risk patient OR long-term care OR home OR skilled<td>Task2f results below

<tr><td>Susceptibility of populations<td>susceptibility OR susceptible<td>Task2g results below

<tr><td>Public health mitigation measures that could be effective for control<td>mitigation OR containment OR control<td>Task2h results below

</table>



"""



h = display(HTML(htmlTaskDetails))
htmlresults="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>

<hr><a name="task2a"><b>Task2a Kaggle Prompt:</b> Data on potential risks factors: Smoking, pre-existing pulmonary disease</a><p><b>Results:</b><p>

Searching for (smoking OR pulmonary OR lung OR COPD OR asthma) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/L/LU/Lung">Lung</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/L/LU/Lung Diseases">Lung Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/R/RE/Respiratory Tract Diseases">Respiratory Tract Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/R/RH/Rhinovirus">Rhinovirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/L/LU/Lung Transplantation">Lung Transplantation</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=5><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/L/LU/Lung">Lung</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32216961">

Imaging manifestations and diagnostic value of chest CT of coronavirus disease 2019 (COVID-19) in the Xiaogan area.

</a>

<small>(PMID32216961</small>)

<br>...RESULTS: Chest CT revealed  abnormal <b>lung</b> shadows in 110 patients.

<td>Journal Article</td>

<td>2020/05</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32100485">

Chest Radiographic and CT Findings of the 2019 Novel Coronavirus Disease (COVID-19): Analysis of Nine Patients Treated in Korea.

</a>

<small>(PMID32100485</small>)

<br>...Fisher's exact test was used to compare CT findings depending on the shape of <b>pulmonary</b> lesions...In total, 77 <b>pulmonary</b> lesions were found, including patchy lesions (39%), large confluent lesions (13%), and small nodular lesions (48%)...The peripheral and posterior <b>lung</b> fields were involved in 78% and 67% of the lesions, respectively.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32109443">

Clinical and computed tomographic imaging features of novel coronavirus pneumonia caused by SARS-CoV-2.

</a>

<small>(PMID32109443</small>)

<br>...The lesion was primarily located in the peripheral area under the pleura with possible extension towards the <b>pulmonary</b> hilum.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32191764">

Association of radiologic findings with mortality of patients infected with 2019  novel coronavirus in Wuhan, China.

</a>

<small>(PMID32191764</small>)

<br>...CT findings of NCIP were featured by predominant ground glass opacities mixed with consolidations, mainly peripheral or combined peripheral and central distributions, bilateral and lower <b>lung</b> zones being mostly involved.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=13><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32107577">

Imaging and clinical features of patients with 2019 novel coronavirus SARS-CoV-2.

</a>

<small>(PMID32107577</small>)

<br>...We analyzed the clinical characteristics of the patients, as well as the distribution characteristics, pattern, morphology, and accompanying manifestations of <b>lung</b> lesions...More than half of the patients presented bilateral, multifocal <b>lung</b> lesions, with peripheral distribution, and 53 (59%) patients had more than two lobes involved.

<td>Journal Article</td>

<td>2020/05</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32088847">

(18)F-FDG PET/CT findings of COVID-19: a series of four highly suspected cases.

</a>

<small>(PMID32088847</small>)

<br>...RESULTS: All patients had peripheral ground-glass opacities and/or lung consolidations in more than two <b>pulmonary</b> lobes...<b>Lung</b> lesions were characterized by a high (18)F-FDG uptake and there was evidence of lymph node involvement...Conversely, disseminated disease was absent, a finding suggesting that COVID-19 has <b>pulmonary</b> tropism.

<td>Case Reports; Journal Article</td>

<td>2020/05</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32112884">

Clinical characteristics and imaging manifestations of the 2019 novel coronavirus disease (COVID-19):A multi-center study in Wenzhou city, Zhejiang, China.

</a>

<small>(PMID32112884</small>)

<br>...On chest computed tomography (CT), <b>lung</b> segments 6 and 10 were mostly involved...Lesions were more localized in the peripheral <b>lung</b> with a patchy form...The imaging pattern of multifocal peripheral ground glass or mixed opacity with predominance in the lower <b>lung</b> is highly suspicious of COVID-19 in the first week of disease onset.

<td>Journal Article; Multicenter Study</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32240123">

Preliminary Estimates of the Prevalence of Selected Underlying Health Conditions  Among Patients with Coronavirus Disease 2019 - United States, February 12-March 28, 2020.

</a>

<small>(PMID32240123</small>)

<br>...The most commonly reported conditions were diabetes mellitus, chronic <b>lung</b> disease, and cardiovascular disease.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32252784">

CT imaging changes of corona virus disease 2019(COVID-19): a multi-center study in Southwest China.

</a>

<small>(PMID32252784</small>)

<br>...Most of the lesions identified in chest CT images were multiple lesions of bilateral lungs, lesions were more localized in the peripheral <b>lung</b>, 109 (83%) patients had more than two lobes involved, 20 (15%) patients presented with patchy ground glass opacities, patchy ground glass opacities and consolidation of lesions co-existing in 61 (47%) cases.

<td>Journal Article; Multicenter Study; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32048163">

Clinical and biochemical indexes from 2019-nCoV infected patients linked to viral loads and lung injury.

</a>

<small>(PMID32048163</small>)

<br>...The viral load of 2019-nCoV detected from patient respiratory tracts was positively linked to <b>lung</b> disease severity...ALB, LYM, LYM (%), LDH, NEU (%), and CRP were highly correlated to the  acute <b>lung</b> injury...Age, viral load, <b>lung</b> injury score, and blood biochemistry indexes, albumin (ALB), CRP, LDH, LYM (%), LYM, and NEU (%), may be predictors of disease severity...Moreover, the Angiotensin II level in the plasma sample from 2019-nCoV infected patients was markedly elevated and linearly associated to viral load and <b>lung</b> injury.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32054787">

Prophylactic and therapeutic remdesivir (GS-5734) treatment in the rhesus macaque model of MERS-CoV infection.

</a>

<small>(PMID32054787</small>)

<br>...Prophylactic remdesivir treatment initiated 24 h prior to inoculation completely prevented MERS-CoV-induced clinical disease, strongly inhibited MERS-CoV replication in respiratory tissues, and prevented the formation of <b>lung</b> lesions...Therapeutic remdesivir treatment initiated 12 h postinoculation also provided a clear clinical benefit, with a reduction in clinical signs, reduced virus replication in the lungs, and decreased presence and severity of <b>lung</b> lesions.

<td>Journal Article; Research Support, N.I.H., Intramural</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32043983">

Clinical evidence does not support corticosteroid treatment for 2019-nCoV lung injury.

</a>

<small>(PMID32043983</small>)

<br>....

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31924756">

Comparative therapeutic efficacy of remdesivir and combination lopinavir, ritonavir, and interferon beta against MERS-CoV.

</a>

<small>(PMID31924756</small>)

<br>...In mice, both prophylactic and therapeutic RDV improve <b>pulmonary</b> function and reduce lung viral loads and severe lung pathology...Therapeutic LPV/RTV-IFNb improves <b>pulmonary</b> function but does not reduce virus replication or severe lung  pathology.

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32020836">

RNA based mNGS approach identifies a novel human coronavirus from two individual  pneumonia cases in 2019 Wuhan outbreak.

</a>

<small>(PMID32020836</small>)

<br>...The two patients shared common clinical features including fever, cough, and multiple ground-glass opacities in the bilateral <b>lung</b> field with patchy infiltration.

<td>Case Reports; Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32172669">

Diagnosis and clinical management of severe acute respiratory syndrome Coronavirus 2 (SARS-CoV-2) infection: an operational recommendation of Peking Union Medical College Hospital (V2.0).

</a>

<small>(PMID32172669</small>)

<br>...The clinical features include fever, coughing, shortness of breath, and inflammatory <b>lung</b> infiltration.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31668197">

The Middle East Respiratory Syndrome (MERS).

</a>

<small>(PMID31668197</small>)

<br>...The elderly, immunocompromised, and those with chronic comorbid liver, <b>lung</b>, and hepatic conditions have a high mortality rate.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/L/LU/Lung Diseases">Lung Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31548072">

Effect of antibiotic treatment in preweaned Holstein calves after experimental bacterial challenge with Pasteurella multocida.

</a>

<small>(PMID31548072</small>)

<br>...The primary objective of this randomized controlled challenge study was to investigate the effect of ampicillin on ultrasonographic (US) <b>lung</b> consolidation  after experimental challenge with Pasteurella multocida in preweaned dairy calves...multocida in <b>lung</b>  tissue at postmortem exam (PME)...<b>Lung</b> US and respiratory scoring were performed 2, 6, 12, and 24 h post-challenge, then US once daily and respiratory scoring twice daily until d 14...once daily for 3 d] when >/=1 cm(2) of <b>lung</b> consolidation was observed and >/=6 h had elapsed since challenge...<b>Lung</b> lesions >/=1 cm(2) were considered positive for consolidation...Gross <b>lung</b> lesions and pathogens were quantified following PME...On d 14, 70% (12 out of 17) of TX and 100% (11 out of 11) of CON calves had <b>lung</b> consolidation, and 24% (4 out of 17) of TX and 27% (3 out of 11) of CON calves had clinical respiratory disease...<b>Lung</b> cultures were positive for P...<b>Lung</b> health benefited from a 3-d ampicillin therapy, but benefits were short-lived...Treatment failures might be due to incomplete resolution of the initial <b>lung</b> infection...Future studies are needed to optimize TX strategies to improve long-term <b>lung</b> health..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/R/RE/Respiratory Tract Diseases">Respiratory Tract Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31790139">

Pathological consequences of the unfolded protein response and downstream protein disulphide isomerases in pulmonary viral infection and disease.

</a>

<small>(PMID31790139</small>)

<br>....

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/R/RH/Rhinovirus">Rhinovirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31513433">

Antiviral immunity is impaired in COPD patients with frequent exacerbations.

</a>

<small>(PMID31513433</small>)

<br>...Patients with frequent exacerbations represent a chronic obstructive <b>pulmonary</b> disease (COPD) subgroup requiring better treatment options...The aim of this study was to determine the innate immune mechanisms that underlie susceptibility to frequent exacerbations in <b>COPD</b>...We measured sputum expression of immune mediators and bacterial loads in samples from patients with <b>COPD</b> at stable state and during virus-associated exacerbations...In vitro immune responses to rhinovirus infection in differentiated primary bronchial epithelial cells (BECs) sampled from patients with <b>COPD</b> were additionally evaluated...These data implicate deficient airway innate immunity involving epithelial cells in the increased propensity to exacerbations observed in some patients with <b>COPD</b>...Therapeutic approaches to boost innate antimicrobial immunity in the <b>lung</b> could be a viable strategy for prevention and treatment of frequent exacerbations..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=smoking+OR+pulmonary+OR+lung+OR+COPD+OR+asthma&from=CORD19#/L/LU/Lung Transplantation">Lung Transplantation</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31761338">

Multiday maintenance of extracorporeal lungs using cross-circulation with conscious swine.

</a>

<small>(PMID31761338</small>)

<br>...OBJECTIVES: <b>Lung</b> remains the least-utilized solid organ for transplantation...Here, we demonstrate the feasibility of extending normothermic extracorporeal <b>lung</b> support to 4 days using cross-circulation with conscious swine...RESULTS: Throughout 4 days of normothermic support, extracorporeal <b>lung</b> function was maintained (arterial oxygen tension/inspired oxygen fraction >400 mm Hg; compliance >20 mL/cm H2O), and recipient swine were hemodynamically stable (lactate <3 mmol/L; pH, 7.42 +/-  0.05).

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2020/04</td>

</tr>

</table>

<p>There are also 2308 matches before 2019/12

<hr><a name="task2b"><b>Task2b Kaggle Prompt:</b> Data on potential risks factors: Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities</a><p><b>Results:</b><p>

Searching for (co-infection OR comorbidity  OR comorbidities OR  co-morbidities OR co-morbidity) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=co-infection+OR+comorbidity++OR+comorbidities+OR++co-morbidities+OR+co-morbidity&from=CORD19#/R/RE/Respiratory Tract Infections">Respiratory Tract Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=co-infection+OR+comorbidity++OR+comorbidities+OR++co-morbidities+OR+co-morbidity&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=co-infection+OR+comorbidity++OR+comorbidities+OR++co-morbidities+OR+co-morbidity&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=co-infection+OR+comorbidity++OR+comorbidities+OR++co-morbidities+OR+co-morbidity&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=co-infection+OR+comorbidity++OR+comorbidities+OR++co-morbidities+OR+co-morbidity&from=CORD19#/C/CA/Cat Diseases">Cat Diseases</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=co-infection+OR+comorbidity++OR+comorbidities+OR++co-morbidities+OR+co-morbidity&from=CORD19#/R/RE/Respiratory Tract Infections">Respiratory Tract Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31004768">

Respiratory virus infection among hospitalized adult patients with or without clinically apparent respiratory infection: a prospective cohort study.

</a>

<small>(PMID31004768</small>)

<br>...Respiratory viruses were detected in 34.1% (73/214) of patients, and <b>co-infection</b> occurred in 7.9% (17/214) of patients.

<td>Comparative Study; Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=co-infection+OR+comorbidity++OR+comorbidities+OR++co-morbidities+OR+co-morbidity&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32186278">

Influenza-associated pneumonia as reference to assess seriousness of coronavirus  disease (COVID-19).

</a>

<small>(PMID32186278</small>)

<br>...There were more case fatalities among COVID-19 patients without <b>comorbidities</b> than in the reference cohort.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=co-infection+OR+comorbidity++OR+comorbidities+OR++co-morbidities+OR+co-morbidity&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32007143">

Epidemiological and clinical characteristics of 99 cases of 2019 novel coronavirus pneumonia in Wuhan, China: a descriptive study.

</a>

<small>(PMID32007143</small>)

<br>...INTERPRETATION: The 2019-nCoV infection was of clustering onset, is more likely to affect older males with <b>comorbidities</b>, and can result in severe and even fatal respiratory diseases such as acute respiratory distress syndrome.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32191764">

Association of radiologic findings with mortality of patients infected with 2019  novel coronavirus in Wuhan, China.

</a>

<small>(PMID32191764</small>)

<br>... The <b>comorbidity</b> rate in mortality group was significantly higher than in survival group (80% vs 29%, P = 0.018)...2019-nCoV was more likely to infect elderly people with  chronic <b>comorbidities</b>.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=co-infection+OR+comorbidity++OR+comorbidities+OR++co-morbidities+OR+co-morbidity&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32166607">

A Review of Coronavirus Disease-2019 (COVID-19).

</a>

<small>(PMID32166607</small>)

<br>...The disease is mild in most people; in some (usually the elderly and those with <b>comorbidities</b>), it may progress to pneumonia, acute respiratory distress syndrome (ARDS) and multi organ dysfunction.

<td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=co-infection+OR+comorbidity++OR+comorbidities+OR++co-morbidities+OR+co-morbidity&from=CORD19#/C/CA/Cat Diseases">Cat Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31822209">

Co-infection with Bartonella henselae and Sarcocystis sp. in a 6-year-old male neutered domestic longhair cat with progressive multifocal neurological signs.

</a>

<small>(PMID31822209</small>)

<br>....

<td>Case Reports; Journal Article</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 415 matches before 2019/12

<hr><a name="task2c"><b>Task2c Kaggle Prompt:</b> Data on potential risks factors: Neonates and pregnant women</a><p><b>Results:</b><p>

Searching for (pregnancy OR pregnant OR neonate OR newborn OR gestation OR fetus) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=pregnancy+OR+pregnant+OR+neonate+OR+newborn+OR+gestation+OR+fetus&from=CORD19#/S/SW/Swine Diseases">Swine Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=pregnancy+OR+pregnant+OR+neonate+OR+newborn+OR+gestation+OR+fetus&from=CORD19#/C/CA/Cattle">Cattle</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=pregnancy+OR+pregnant+OR+neonate+OR+newborn+OR+gestation+OR+fetus&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=pregnancy+OR+pregnant+OR+neonate+OR+newborn+OR+gestation+OR+fetus&from=CORD19#/D/DI/Diestrus">Diestrus</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=pregnancy+OR+pregnant+OR+neonate+OR+newborn+OR+gestation+OR+fetus&from=CORD19#/S/SW/Swine Diseases">Swine Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31805938">

High levels of unreported intraspecific diversity among RNA viruses in faeces of  neonatal piglets with diarrhoea.

</a>

<small>(PMID31805938</small>)

<br>...BACKGROUND: Diarrhoea is a major cause of death in <b>neonate</b> pigs and most of the viruses that cause it are RNA viruses...CONCLUSIONS: Among the cases analysed, Rotaviruses were the main aetiological agents of diarrhoea in <b>neonate</b> pigs.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=pregnancy+OR+pregnant+OR+neonate+OR+newborn+OR+gestation+OR+fetus&from=CORD19#/C/CA/Cattle">Cattle</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31677831">

Associations between maternal characteristics and health, survival, and performance of dairy heifers from birth through first lactation.

</a>

<small>(PMID31677831</small>)

<br>...Holstein heifers (n = 1,811) derived from artificial insemination were categorized as (1) daughters of primiparous cows that, consequently, were nonlactating heifers during <b>gestation</b> (Prim-NoL; n  = 787); (2) daughters of multiparous cows that did not have any clinical diseases in the previous lactation (Mult-NoCD; n = 638); and (3) daughters of multiparous  cows that had at least one clinical disease in the previous lactation (Mult-CD; n = 386)...28%) and  to lose <b>pregnancy</b> as a heifer (9 vs.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=pregnancy+OR+pregnant+OR+neonate+OR+newborn+OR+gestation+OR+fetus&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31767069">

Attenuation and characterization of porcine enteric alphacoronavirus strain GDS04 via serial cell passage.

</a>

<small>(PMID31767069</small>)

<br>...Porcine enteric alphacoronavirus (PEAV) is a newly identified swine enteropathogenic coronavirus that causes watery diarrhea in <b>newborn</b> piglets...<b>Newborn</b> piglets were orally inoculated with PEAV P15, P67 and  P100...Importantly, all P100-inoculated <b>newborn</b> piglets survived, indicating that P100 was an attenuated variant.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=pregnancy+OR+pregnant+OR+neonate+OR+newborn+OR+gestation+OR+fetus&from=CORD19#/D/DI/Diestrus">Diestrus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31541785">

Triplex doppler ultrasonography to describe the uterine arteries during diestrus  and progesterone profile in pregnant and non-pregnant bitches of different sizes.

</a>

<small>(PMID31541785</small>)

<br>...Hemodynamics of uterine vascularization is modified throughout <b>pregnancy</b> to meet  the increasing demand of the growing fetuses and triplex doppler ultrasonography  is widely used in human medicine to study the uterine arteries and assess the fetal and placental conditions...Thirty-three out of forty-four  bitches were <b>pregnant</b>, including 6 abnormal pregnancies (resorption of more than  10% of the embryos)...We observed that RI and PI decreased over time and were significantly lower for <b>pregnant</b> bitches compared to non-pregnant ones from 30 days post-ovulation...The only significant difference between <b>pregnant</b> and non-pregnant bitches was observed from 30 days post-ovulation..

<td>Journal Article</td>

<td>2020/01</td>

</tr>

</table>

<p>There are also 579 matches before 2019/12

<hr><a name="task2d"><b>Task2d Kaggle Prompt:</b> Data on potential risks factors: Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.</a><p><b>Results:</b><p>

Searching for (economic OR socio-economic OR socioeconomic OR behavioral OR poverty) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=economic+OR+socio-economic+OR+socioeconomic+OR+behavioral+OR+poverty&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=economic+OR+socio-economic+OR+socioeconomic+OR+behavioral+OR+poverty&from=CORD19#/P/PO/Porcine respiratory and reproductive syndrome virus">Porcine respiratory and reproductive syndrome virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=economic+OR+socio-economic+OR+socioeconomic+OR+behavioral+OR+poverty&from=CORD19#/C/CH/Cholera">Cholera</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=economic+OR+socio-economic+OR+socioeconomic+OR+behavioral+OR+poverty&from=CORD19#/E/EX/Extracorporeal Circulation">Extracorporeal Circulation</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=7><a href="http://www.softconcourse.com/CORD19/?filterText=economic+OR+socio-economic+OR+socioeconomic+OR+behavioral+OR+poverty&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32238182">

The fiscal value of human lives lost from coronavirus disease (COVID-19) in China.

</a>

<small>(PMID32238182</small>)

<br>...Re-estimation of the <b>economic</b> model alternately with 5% and 10 discount rates led to a reduction in the expected total fiscal value by 21.3% and 50.4%, respectively...Furthermore, the re-estimation of the <b>economic</b> model using the world's highest average life expectancy of 87.1 years (which is that of Japanese  females), instead of the national life expectancy of 76.4 years, increased the total fiscal value by Int$ 229,456,430 (24.8%)..

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32186277">

Rapidly increasing cumulative incidence of coronavirus disease (COVID-19) in the  European Union/European Economic Area and the United Kingdom, 1 January to 15 March 2020.

</a>

<small>(PMID32186277</small>)

<br>...The cumulative incidence of coronavirus disease (COVID-19) cases is showing similar trends in European Union/European <b>Economic</b> Area countries and the United  Kingdom confirming that, while at a different stage depending on the country, the COVID-19 pandemic is progressing rapidly in all countries.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32156332">

Potential scenarios for the progression of a COVID-19 epidemic in the European Union and the European Economic Area, March 2020.

</a>

<small>(PMID32156332</small>)

<br>...Two months after the emergence of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the possibility of established and widespread community transmission in the European Union and European <b>Economic</b> Area (EU/EEA) is becoming more likely.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32046815">

Laboratory readiness and response for novel coronavirus (2019-nCoV) in expert laboratories in 30 EU/EEA countries, January 2020.

</a>

<small>(PMID32046815</small>)

<br>...We assessed the required expertise and capacity for molecular detection of 2019-nCoV in specialised laboratories in 30 European Union/European <b>Economic</b> Area (EU/EEA) countries.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31936476">

Isolation and Identification of Porcine Deltacoronavirus and Alteration of Immunoglobulin Transport Receptors in the Intestinal Mucosa of PDCoV-Infected Piglets.

</a>

<small>(PMID31936476</small>)

<br>...Porcine deltacoronavirus (PDCoV) is a porcine enteropathogenic coronavirus that causes watery diarrhea, vomiting, and frequently death in piglets, causing serious <b>economic</b> losses to the pig industry.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226294">

Knowledge, attitudes, and practices towards COVID-19 among Chinese residents during the rapid rise period of the COVID-19 outbreak: a quick online cross-sectional survey.

</a>

<small>(PMID32226294</small>)

<br>...Most Chinese residents of  a relatively high <b>socioeconomic</b> status, in particular women, are knowledgeable about COVID-19, hold optimistic attitudes, and have appropriate practices towards COVID-19...Due to the limited sample representativeness, we must be cautious when generalizing these findings to populations of a low <b>socioeconomic</b> status..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=economic+OR+socio-economic+OR+socioeconomic+OR+behavioral+OR+poverty&from=CORD19#/P/PO/Porcine respiratory and reproductive syndrome virus">Porcine respiratory and reproductive syndrome virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31767072">

S100A9 regulates porcine reproductive and respiratory syndrome virus replication  by interacting with the viral nucleocapsid protein.

</a>

<small>(PMID31767072</small>)

<br>...Porcine reproductive and respiratory syndrome virus (PRRSV) has caused huge <b>economic</b> losses to the pig industry worldwide over the last 30 years, yet the associated viral-host interactions remain poorly understood.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=economic+OR+socio-economic+OR+socioeconomic+OR+behavioral+OR+poverty&from=CORD19#/C/CH/Cholera">Cholera</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31905206">

Risk perception and behavioral change during epidemics: Comparing models of individual and collective learning.

</a>

<small>(PMID31905206</small>)

<br>....

<td>Comparative Study; Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=economic+OR+socio-economic+OR+socioeconomic+OR+behavioral+OR+poverty&from=CORD19#/E/EX/Extracorporeal Circulation">Extracorporeal Circulation</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31761338">

Multiday maintenance of extracorporeal lungs using cross-circulation with conscious swine.

</a>

<small>(PMID31761338</small>)

<br>...METHODS: A swine <b>behavioral</b> training program and custom enclosure were developed to enable multiday cross-circulation  between extracorporeal lungs and recipient swine.

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2020/04</td>

</tr>

</table>

<p>There are also 1086 matches before 2019/12

<hr><a name="task2e"><b>Task2e Kaggle Prompt:</b> Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors</a><p><b>Results:</b><p>

Searching for (basic reproductive number OR  incubation OR serial OR transmission OR environment OR environmental OR environs OR surroundings) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=basic+reproductive+number+OR++incubation+OR+serial+OR+transmission+OR+environment+OR+environmental+OR+environs+OR+surroundings&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=basic+reproductive+number+OR++incubation+OR+serial+OR+transmission+OR+environment+OR+environmental+OR+environs+OR+surroundings&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31995857">

Early Transmission Dynamics in Wuhan, China, of Novel Coronavirus-Infected Pneumonia.

</a>

<small>(PMID31995857</small>)

<br>...In the early period of exponential growth, we estimated the epidemic doubling time and the <b>basic</b> <b>reproductive</b> <b>number</b>...With a mean serial interval  of 7.5 days (95% CI, 5.3 to 19), the <b>basic</b> <b>reproductive</b> <b>number</b> was estimated to be 2.2 (95% CI, 1.4 to 3.9).

<td>Journal Article; Research Support, N.I.H., Extramural; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

</table>

<p>There are also 61 matches before 2019/12

<hr><a name="task2f"><b>Task2f Kaggle Prompt:</b> Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups</a><p><b>Results:</b><p>

Searching for (severity OR fatality AND hospitalized OR high-risk patient OR long-term care OR home OR skilled) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/R/RE/Respiratory Tract Infections">Respiratory Tract Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/Z/ZO/Zoonoses">Zoonoses</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/C/CR/Croup">Croup</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/L/LE/Lectins, C-Type">Lectins, C-Type</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/R/RE/Respiratory Tract Infections">Respiratory Tract Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31992388">

Real-time tentative assessment of the epidemiological characteristics of novel coronavirus infections in Wuhan, China, as at 22 January 2020.

</a>

<small>(PMID31992388</small>)

<br>...While the overall <b>severity</b> profile among cases may change as more mild cases are identified, we estimate a risk of fatality among hospitalised cases at 14% (95% confidence interval: 3.9-32%)..

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32186278">

Influenza-associated pneumonia as reference to assess seriousness of coronavirus  disease (COVID-19).

</a>

<small>(PMID32186278</small>)

<br>...Information on <b>severity</b> of coronavirus disease (COVID-19) (transmissibility, disease seriousness, impact) is crucial for preparation of healthcare sectors.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=8><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32216961">

Imaging manifestations and diagnostic value of chest CT of coronavirus disease 2019 (COVID-19) in the Xiaogan area.

</a>

<small>(PMID32216961</small>)

<br>...Data were gathered regarding the  presence of chest computed tomography (CT) abnormalities; the distribution, morphology, density, location, and stage of abnormal shadows on chest CT; and observing the correlation between the <b>severity</b> of chest infection and lymphocyte  ratio and blood oxygen saturation (SPO2) in patients.

<td>Journal Article</td>

<td>2020/05</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32092539">

Is COVID-19 receiving ADE from other coronaviruses?

</a>

<small>(PMID32092539</small>)

<br>...One of the most perplexing questions regarding the current COVID-19 coronavirus epidemic is the discrepancy between the <b>severity</b> of cases observed in the Hubei province of China and those occurring elsewhere in the world.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32048163">

Clinical and biochemical indexes from 2019-nCoV infected patients linked to viral loads and lung injury.

</a>

<small>(PMID32048163</small>)

<br>...Here we report the epidemiological, clinical, laboratory, and radiological characteristics, as well as potential biomarkers for predicting disease <b>severity</b> in 2019-nCoV-infected patients in Shenzhen, China...The viral load of 2019-nCoV detected from patient respiratory tracts was positively linked to lung disease <b>severity</b>...Age, viral load, lung injury score, and blood biochemistry indexes, albumin (ALB), CRP, LDH, LYM (%), LYM, and NEU (%), may be predictors of disease <b>severity</b>.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32054787">

Prophylactic and therapeutic remdesivir (GS-5734) treatment in the rhesus macaque model of MERS-CoV infection.

</a>

<small>(PMID32054787</small>)

<br>...Therapeutic remdesivir treatment initiated 12 h postinoculation also provided a clear clinical benefit, with a reduction in clinical signs, reduced virus replication in the lungs, and decreased presence and <b>severity</b> of lung lesions.

<td>Journal Article; Research Support, N.I.H., Intramural</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32202646">

Factors Associated With Mental Health Outcomes Among Health Care Workers Exposed  to Coronavirus Disease 2019.

</a>

<small>(PMID32202646</small>)

<br>...Main Outcomes and Measures: The degree of symptoms of depression, anxiety, insomnia, and distress was assessed by the Chinese versions of the 9-item Patient Health Questionnaire, the 7-item Generalized Anxiety Disorder scale, the 7-item Insomnia <b>Severity</b> Index, and the 22-item Impact of Event Scale-Revised, respectively...Nurses, women, frontline health care workers, and those working in Wuhan, China, reported more severe degrees of all measurements of mental health symptoms than other health care workers (eg, median [IQR] Patient Health Questionnaire scores among physicians vs nurses: 4.0 [1.0-7.0] vs 5.0 [2.0-8.0]; P = .007; median [interquartile range {IQR}] Generalized Anxiety Disorder scale scores among men vs women: 2.0 [0-6.0] vs 4.0  [1.0-7.0]; P < .001; median [IQR] Insomnia <b>Severity</b> Index scores among frontline  vs second-line workers: 6.0 [2.0-11.0] vs 4.0 [1.0-8.0]; P < .001; median [IQR] Impact of Event Scale-Revised scores among those in Wuhan vs those in Hubei outside Wuhan and those outside Hubei: 21.0 [8.5-34.5] vs 18.0 [6.0-28.0] in Hubei outside Wuhan and 15.0 [4.0-26.0] outside Hubei; P < .001).

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32102625">

Detectable 2019-nCoV viral RNA in blood is a strong indicator for the further clinical severity.

</a>

<small>(PMID32102625</small>)

<br>...Importantly, all of the 6 patients with detectable viral RNA in the blood cohort progressed to severe symptom stage, indicating a strong correlation  of serum viral RNA with the disease <b>severity</b> (p-value = 0.0001).

<td>Letter</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226295">

COVID-19: what has been learned and to be learned about the novel coronavirus disease.

</a>

<small>(PMID32226295</small>)

<br>...However, COVID-19 has lower <b>severity</b> and mortality than SARS but is much more transmissive and affects more elderly individuals than youth and more men than women.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/Z/ZO/Zoonoses">Zoonoses</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226286">

Zoonotic origins of human coronaviruses.

</a>

<small>(PMID32226286</small>)

<br>...In addition, the requirements for successful host switches and the implications of virus evolution on disease <b>severity</b> are also highlighted..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/C/CR/Croup">Croup</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31542653">

Defining atypical croup: A case report and review of the literature.

</a>

<small>(PMID31542653</small>)

<br>...A variety of definitions of  atypical croup were identified based on recurrence, duration of symptoms, <b>severity</b>, and etiology.

<td>Case Reports; Journal Article; Systematic Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=severity+OR+fatality+AND+hospitalized+OR+high-risk+patient+OR+long-term+care+OR+home+OR+skilled&from=CORD19#/L/LE/Lectins, C-Type">Lectins, C-Type</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32152943">

CLEC5A: A Promiscuous Pattern Recognition Receptor to Microbes and Beyond.

</a>

<small>(PMID32152943</small>)

<br>...For example, in vivo studies in mouse models have demonstrated that CLEC5A is responsible for flaviviruses-induced hemorrhagic shock and neuroinflammation, and a CLEC5A polymorphism in humans is associated with disease <b>severity</b> following infection with dengue virus.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

</table>

<p>There are also 1061 matches before 2019/12

<hr><a name="task2g"><b>Task2g Kaggle Prompt:</b> Susceptibility of populations</a><p><b>Results:</b><p>

Searching for (susceptibility OR susceptible) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=susceptibility+OR+susceptible&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=susceptibility+OR+susceptible&from=CORD19#/M/MA/Macrophages">Macrophages</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=susceptibility+OR+susceptible&from=CORD19#/P/PI/Picornaviridae Infections">Picornaviridae Infections</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=6><a href="http://www.softconcourse.com/CORD19/?filterText=susceptibility+OR+susceptible&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32112886">

Characteristics of COVID-19 infection in Beijing.

</a>

<small>(PMID32112886</small>)

<br>...Population was generally <b>susceptible</b>, and with a relatively low fatality rate.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32240128">

Asymptomatic and Presymptomatic SARS-CoV-2 Infections in Residents of a Long-Term Care Skilled Nursing Facility - King County, Washington, March 2020.

</a>

<small>(PMID32240128</small>)

<br>...Older adults are <b>susceptible</b> to severe coronavirus disease 2019 (COVID-19) outcomes as a consequence of their age and, in some cases, underlying health conditions (1).

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32188484">

Host susceptibility to severe COVID-19 and establishment of a host risk score: findings of 487 cases outside Wuhan.

</a>

<small>(PMID32188484</small>)

<br>....

<td>Letter; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32094336">

High expression of ACE2 receptor of 2019-nCoV on the epithelial cells of oral mucosa.

</a>

<small>(PMID32094336</small>)

<br>...Preliminarily, those findings have explained the basic mechanism that the oral cavity is a potentially high risk for 2019-nCoV infectious <b>susceptibility</b> and provided a piece of evidence for the future prevention strategy in dental clinical practice as well as daily life..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32111262">

A mathematical model for simulating the phase-based transmissibility of a novel coronavirus.

</a>

<small>(PMID32111262</small>)

<br>...RESULTS: The value of R0  was estimated of 2.30 from reservoir to person and 3.58 from person to person which means that the expected number of secondary infections that result from introducing a single infected individual into an otherwise <b>susceptible</b> population was 3.58.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=susceptibility+OR+susceptible&from=CORD19#/M/MA/Macrophages">Macrophages</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31825972">

IL-4/IL-13 polarization of macrophages enhances Ebola virus glycoprotein-dependent infection.

</a>

<small>(PMID31825972</small>)

<br>...A critical step towards  development of effective therapeutics is a better understanding of factors that govern host <b>susceptibility</b> to this pathogen...Macrophages polarized towards a M2-like anti-inflammatory state by combined IL-4  and IL-13 treatment were more <b>susceptible</b> to rVSV/EBOV GP, but not to wild-type VSV (rVSV/G), suggesting that EBOV GP-dependent entry events were enhanced by these cytokines...Our findings provide an increased understanding of the host factors in macrophages governing <b>susceptibility</b> to filoviruses and identify novel murine receptors mediating EBOV entry..

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=susceptibility+OR+susceptible&from=CORD19#/P/PI/Picornaviridae Infections">Picornaviridae Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31513433">

Antiviral immunity is impaired in COPD patients with frequent exacerbations.

</a>

<small>(PMID31513433</small>)

<br>...The aim of this study was to determine the innate immune mechanisms that underlie <b>susceptibility</b> to frequent exacerbations in COPD.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 1693 matches before 2019/12

<hr><a name="task2h"><b>Task2h Kaggle Prompt:</b> Public health mitigation measures that could be effective for control</a><p><b>Results:</b><p>

Searching for (mitigation OR containment OR control) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/S/SE/Severe Acute Respiratory Syndrome">Severe Acute Respiratory Syndrome</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/C/CA/Cattle Diseases">Cattle Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/V/VI/Virus Diseases">Virus Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/I/IN/Infection Control">Infection Control</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/P/PO/Porcine respiratory and reproductive syndrome virus">Porcine respiratory and reproductive syndrome virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/T/TU/Tuberculosis">Tuberculosis</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/P/PO/Polymerase Chain Reaction">Polymerase Chain Reaction</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/P/PL/Plant Extracts">Plant Extracts</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/H/HE/Henipavirus Infections">Henipavirus Infections</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=34><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32119825">

Feasibility of controlling COVID-19 outbreaks by isolation of cases and contacts.

</a>

<small>(PMID32119825</small>)

<br>...BACKGROUND: Isolation of cases and contact tracing is used to <b>control</b> outbreaks of infectious diseases, and has been used for coronavirus disease 2019 (COVID-19)...Whether this strategy will achieve <b>control</b> depends on characteristics of both the pathogen and the response...Here we use a mathematical model to assess if isolation and contact tracing are able to <b>control</b> onwards transmission from imported cases of COVID-19...To <b>control</b> the majority of outbreaks, for R0 of 2.5 more than 70% of contacts had to be traced, and for an R0 of 3.5 more than 90% of contacts had to  be traced...INTERPRETATION: In most scenarios, highly effective contact tracing and case isolation is enough to <b>control</b> a new outbreak of COVID-19 within 3 months...The probability of <b>control</b> decreases with long delays from symptom onset to isolation, fewer cases ascertained by contact tracing, and increasing transmission before symptoms...This model can be modified to reflect updated transmission characteristics and more specific definitions of outbreak <b>control</b> to assess the potential success of local response efforts.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32123989">

The novel coronavirus (SARS-CoV-2) infections in China: prevention, control and challenges.

</a>

<small>(PMID32123989</small>)

<br>....

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32031583">

Molecular Diagnosis of a Novel Coronavirus (2019-nCoV) Causing an Outbreak of Pneumonia.

</a>

<small>(PMID32031583</small>)

<br>...RESULTS: Using RNA extracted from cells infected by SARS coronavirus as a positive <b>control</b>, these assays were shown to have a dynamic range of at least seven orders of magnitude (2x10-4-2000 TCID50/reaction)...All negative <b>control</b> samples were negative in the assays.

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32240095">

Emergence of a Novel Coronavirus (COVID-19): Protocol for Extending Surveillance  Used by the Royal College of General Practitioners Research and Surveillance Centre and Public Health England.

</a>

<small>(PMID32240095</small>)

<br>...With the emergence of the  international outbreak of the coronavirus infection (COVID-19), a UK national approach to <b>containment</b> has been established to test people suspected of exposure to COVID-19...At the same time and separately, the RCGP RSC's surveillance has been extended to monitor the temporal and geographical distribution of COVID-19 infection in the community as well as assess the effectiveness of the <b>containment</b> strategy...OBJECTIVES: The aims of this study are to surveil COVID-19 in both asymptomatic populations and ambulatory cases with respiratory infections, ascertain both the rate and pattern of COVID-19 spread, and assess the effectiveness of the <b>containment</b> policy...(2) Extension of current virological surveillance and testing people with influenza-like illness or lower respiratory tract infections  (LRTI)-with the caveat that people suspected to have or who have been exposed to  COVID-19 should be referred to the national <b>containment</b> pathway and not seen in primary care...CONCLUSIONS: We have rapidly converted the established national RCGP RSC influenza surveillance system into one that can test the effectiveness of the COVID-19 <b>containment</b> policy.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32240096">

Preventive Behaviors Conveyed on YouTube to Mitigate Transmission of COVID-19: Cross-Sectional Study.

</a>

<small>(PMID32240096</small>)

<br>...RESULTS: Fewer than one-third of the videos covered any of the seven key prevention behaviors listed on the US Centers for Disease <b>Control</b> and Prevention website.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32264957">

Fighting against the common enemy of COVID-19: a practice of building a community with a shared future for mankind.

</a>

<small>(PMID32264957</small>)

<br>...To date, we have found it is one of the greatest challenges to human beings in fighting against COVID-19 in the history, because SARS-CoV-2 is different from SARS-CoV and MERS-CoV in terms of biological features and transmissibility, and also found the <b>containment</b> strategies including the non-pharmaceutical public health measures implemented in China are  effective and successful.

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32155431">

Unveiling the Origin and Transmission of 2019-nCoV.

</a>

<small>(PMID32155431</small>)

<br>...Recent studies (Huang et al., Chan et al., and Zhou et al.) have provided timely insights into its origin  and ability to spread among humans, informing infection prevention and <b>control</b> practices..

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32143519">

Prediction of Epidemic Spread of the 2019 Novel Coronavirus Driven by Spring Festival Transportation in China: A Population-Based Study.

</a>

<small>(PMID32143519</small>)

<br>...The results are conducive to monitoring the epidemic prevention and <b>control</b> in various regions..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31995857">

Early Transmission Dynamics in Wuhan, China, of Novel Coronavirus-Infected Pneumonia.

</a>

<small>(PMID31995857</small>)

<br>...Considerable efforts to reduce transmission will be required to <b>control</b> outbreaks if similar dynamics apply elsewhere.

<td>Journal Article; Research Support, N.I.H., Extramural; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32035997">

Persistence of coronaviruses on inanimate surfaces and their inactivation with biocidal agents.

</a>

<small>(PMID32035997</small>)

<br>...As no specific therapies are available for SARS-CoV-2, early <b>containment</b> and prevention of further spread will be crucial to stop the ongoing outbreak and to  control this novel infectious thread..

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32032682">

Pathogenicity and transmissibility of 2019-nCoV-A quick overview and comparison with other emerging viruses.

</a>

<small>(PMID32032682</small>)

<br>...Here, the current knowledge in  2019-nCoV pathogenicity and transmissibility is summarized in comparison with several commonly known emerging viruses, and information urgently needed for a better <b>control</b> of the disease is highlighted..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32234117">

Potential short-term outcome of an uncontrolled COVID-19 epidemic in Lombardy, Italy, February to March 2020.

</a>

<small>(PMID32234117</small>)

<br>...Aggressive <b>containment</b> strategies are required..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32182811">

Reverse Logistics Network Design for Effective Management of Medical Waste in Epidemic Outbreaks: Insights from the Coronavirus Disease 2019 (COVID-19) Outbreak in Wuhan (China).

</a>

<small>(PMID32182811</small>)

<br>...In order to <b>control</b> the spread of an epidemic, the effective management of rapidly increased medical waste through establishing a temporary reverse logistics system is of vital importance.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32183901">

Epidemiology, causes, clinical manifestation and diagnosis, prevention and control of coronavirus disease (COVID-19) during the early outbreak period: a scoping review.

</a>

<small>(PMID32183901</small>)

<br>...In this scoping review, 65 research articles published before 31 January 2020 were analyzed and discussed to better understand the epidemiology, causes, clinical diagnosis, prevention and <b>control</b> of this virus...Research articles initially focused on causes, but over time there was an increase of the articles related to prevention and <b>control</b>...During this early period, published research primarily explored the epidemiology, causes, clinical manifestation and diagnosis, as well  as prevention and <b>control</b> of the novel coronavirus...Although these studies are relevant to <b>control</b> the current public emergency, more high-quality research is needed to provide valid and reliable ways to manage this kind of public health emergency in both the short- and long-term..

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32156332">

Potential scenarios for the progression of a COVID-19 epidemic in the European Union and the European Economic Area, March 2020.

</a>

<small>(PMID32156332</small>)

<br>...We propose actions to prepare for potential <b>mitigation</b> phases and coordinate efforts to protect the health of citizens..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32193905">

First Pediatric Case of Coronavirus Disease 2019 in Korea.

</a>

<small>(PMID32193905</small>)

<br>...Current epidemiologic knowledge suggests that relatively few cases are seen among children, which limits opportunities to address pediatric specific issues on infection <b>control</b> and the children's contribution to viral spread in the community.

<td>Case Reports; Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32111295">

Wuhan novel coronavirus (COVID-19): why global control is challenging?

</a>

<small>(PMID32111295</small>)

<br>....

<td>Editorial</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31978945">

A Novel Coronavirus from Patients with Pneumonia in China, 2019.

</a>

<small>(PMID31978945</small>)

<br>...(Funded by the National Key Research and Development Program of China and the National Major Project for <b>Control</b> and Prevention of Infectious Disease in China.)..

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32040190">

A qualitative study of zoonotic risk factors among rural communities in southern  China.

</a>

<small>(PMID32040190</small>)

<br>...However, the risk factors leading  to emergence are poorly understood, which presents a challenge in developing appropriate <b>mitigation</b> strategies for local communities...Policies and programmes  existing in the communities provide opportunities for zoonotic risk <b>mitigation</b>.

<td>Journal Article; Research Support, N.I.H., Extramural; Research Support, Non-U.S. Gov't; Research Support, U.S. Gov't, Non-P.H.S.</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32053579">

Persons Evaluated for 2019 Novel Coronavirus - United States, January 2020.

</a>

<small>(PMID32053579</small>)

<br>...Health care providers should remain vigilant and adhere to recommended infection prevention and <b>control</b> practices when evaluating patients for possible 2019-nCoV infection (6).

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32035533">

What to do next to control the 2019-nCoV epidemic?

</a>

<small>(PMID32035533</small>)

<br>....

<td>Journal Article; Comment</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31992387">

Detection of 2019 novel coronavirus (2019-nCoV) by real-time RT-PCR.

</a>

<small>(PMID31992387</small>)

<br>...<b>Control</b> material is made available through European Virus Archive - Global (EVAg), a European Union infrastructure project.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32191174">

2019-nCoV: The Identify-Isolate-Inform (3I) Tool Applied to a Novel Emerging Coronavirus.

</a>

<small>(PMID32191174</small>)

<br>...An unusually high volume of domestic and international travel corresponding to the beginning of the 2020 Chinese New Year complicated initial identification and <b>containment</b> of infected persons.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32117569">

Therapeutic strategies in an outbreak scenario to treat the novel coronavirus originating in Wuhan, China.

</a>

<small>(PMID32117569</small>)

<br>...Current efforts are focused on <b>containment</b> and quarantine of infected individuals.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32049206">

Synanthropic rodents as virus reservoirs and transmitters.

</a>

<small>(PMID32049206</small>)

<br>...Comprehensive <b>control</b> and preventive activities should  include actions such as the elimination or reduction of rat and mouse populations, sanitary education, reduction of shelters for the animals, and restriction of the access of rodents to residences, water, and food supplies..

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32023775">

An interim review of the epidemiological characteristics of 2019 novel coronavirus.

</a>

<small>(PMID32023775</small>)

<br>...METHODS: We reviewed the currently available literature to provide up-to-date guidance on <b>control</b> measures to be implemented by public health authorities...However, there remain considerable uncertainties, which should be considered when providing guidance to public health authorities on <b>control</b> measures.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226294">

Knowledge, attitudes, and practices towards COVID-19 among Chinese residents during the rapid rise period of the COVID-19 outbreak: a quick online cross-sectional survey.

</a>

<small>(PMID32226294</small>)

<br>...Unprecedented measures have been adopted to <b>control</b> the rapid spread of the ongoing COVID-19 epidemic in China...People's adherence to <b>control</b> measures is affected by their knowledge, attitudes, and practices (KAP) towards COVID-19.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32231374">

Data-based analysis, modelling and forecasting of the COVID-19 outbreak.

</a>

<small>(PMID32231374</small>)

<br>...Our analysis further reveals a significant decline of the case fatality ratio from January 26 to which various factors may have contributed, such as the severe <b>control</b> measures taken in Hubei, China (e.g.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226289">

Perspectives on therapeutic neutralizing antibodies against the Novel Coronavirus SARS-CoV-2.

</a>

<small>(PMID32226289</small>)

<br>...Although many challenges exist, NAbs still offer a therapeutic option to <b>control</b> the current pandemic and the possible re-emergence of the virus in the future, and their development therefore remains a high priority..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226290">

Targeting the Endocytic Pathway and Autophagy Process as a Novel Therapeutic Strategy in COVID-19.

</a>

<small>(PMID32226290</small>)

<br>...Such  knowledge will provide important clues for <b>control</b> of the ongoing epidemic of SARS-CoV-2 infection and treatment of COVID-19..

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226291">

Progression of Mental Health Services during the COVID-19 Outbreak in China.

</a>

<small>(PMID32226291</small>)

<br>...Psychological crisis intervention plays a pivotal role  in the overall deployment of the disease <b>control</b>.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31668197">

The Middle East Respiratory Syndrome (MERS).

</a>

<small>(PMID31668197</small>)

<br>...Person-to-person spread causes hospital and household outbreaks, and thus improved compliance with internationally recommended infection <b>control</b> protocols and rapid implementation of infection control measures are required..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31463771">

Short hairpin RNAs targeting M and N genes reduce replication of porcine deltacoronavirus in ST cells.

</a>

<small>(PMID31463771</small>)

<br>...Currently, there are no effective treatments or vaccines available to <b>control</b> PDCoV.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=4><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31951818">

Influenza A and B in a cohort of outpatient children and adolescent with influenza like-illness during two consecutive influenza seasons.

</a>

<small>(PMID31951818</small>)

<br>...Yearly vaccination with trivalent or quadrivalent vaccines is the main strategy to <b>control</b> influenza.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31931793">

An evaluation of the Zambia influenza sentinel surveillance system, 2011-2017.

</a>

<small>(PMID31931793</small>)

<br>...METHODS: We used the Centers for Disease <b>Control</b> and Prevention guidelines to evaluate the performance of the influenza surveillance system (ISS) in Zambia during 2011-2017 using 9 attributes: (i) data quality and completeness, (ii) timeliness, (iii) representativeness, (iv) flexibility, (v) simplicity, (vi) acceptability, (vii) stability, (viii) utility, and (ix) sustainability.

<td>Evaluation Study; Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31823763">

Evaluation of the influenza sentinel surveillance system in the Democratic Republic of Congo, 2012-2015.

</a>

<small>(PMID31823763</small>)

<br>...METHODS: We used the Centers for Disease <b>Control</b> and Prevention guidelines to evaluate the performance of the  influenza sentinel surveillance system (ISSS) in DRC during 2012-2015.

<td>Evaluation Study; Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/S/SE/Severe Acute Respiratory Syndrome">Severe Acute Respiratory Syndrome</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31668196">

Severe Acute Respiratory Syndrome: Historical, Epidemiologic, and Clinical Features.

</a>

<small>(PMID31668196</small>)

<br>...Strict infection <b>control</b> procedures with respiratory and contact precautions are essential.

<td>Historical Article; Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31531682">

Saikosaponin C exerts anti-HBV effects by attenuating HNF1alpha and HNF4alpha expression to suppress HBV pgRNA synthesis.

</a>

<small>(PMID31531682</small>)

<br>...METHODS: HepG2.2.15 cells were cultured at 37 in the presence of 1-40 mug/mL of SSc or DMSO as a <b>control</b>.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/C/CA/Cattle Diseases">Cattle Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31548072">

Effect of antibiotic treatment in preweaned Holstein calves after experimental bacterial challenge with Pasteurella multocida.

</a>

<small>(PMID31548072</small>)

<br>...once daily for 3 d] or placebo [n = 11, <b>control</b> (CON), saline, equal volume, i.m.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/V/VI/Virus Diseases">Virus Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31767249">

[What are the determinants of viral outbreaks and is it possible to predict their emergence?]

</a>

<small>(PMID31767249</small>)

<br>...After the introduction phase of a viral disease in a territory or a given population and once the first chains of transmission occur, the spread of the disease or its sustainability are possible if the <b>control</b> measures are not implemented or are not sufficiently effective.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31784255">

[Epidemic and emerging prone-infectious diseases: Lessons learned and ways forward].

</a>

<small>(PMID31784255</small>)

<br>...Massive mobility and reluctance in the populations exposed to epidemic and emerging prone-infectious diseases coupled by a weak health system made disease alert and <b>control</b> measures difficult to implement.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/I/IN/Infection Control">Infection Control</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32070466">

Latest updates on COVID-19 from the European Centre for Disease Prevention and Control.

</a>

<small>(PMID32070466</small>)

<br>....

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/P/PO/Porcine respiratory and reproductive syndrome virus">Porcine respiratory and reproductive syndrome virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31753544">

UV-C irradiation is able to inactivate pathogens found in commercially collected  porcine plasma as demonstrated by swine bioassay.

</a>

<small>(PMID31753544</small>)

<br>...Negative <b>control</b> pigs (group 1) were injected with PBS... Positive <b>control</b> pigs (group 5) were injected with a PCV-2 inoculum.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/T/TU/Tuberculosis">Tuberculosis</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31829147">

Under-reporting of TB cases and associated factors: a case study in China.

</a>

<small>(PMID31829147</small>)

<br>...China, which has the third largest TB epidemic in the world and has developed a reporting system to help with the <b>control</b> and prevention of TB, this study examined its effectiveness in Eastern China...Having an accurate account of the number of national TB  cases is essential to understanding the national and global burden of the disease and in managing TB prevention and <b>control</b> efforts.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/P/PO/Polymerase Chain Reaction">Polymerase Chain Reaction</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31629228">

Thermally stable and uniform DNA amplification with picosecond laser ablated graphene rapid thermal cycling device.

</a>

<small>(PMID31629228</small>)

<br>...Rapid thermal cycling (RTC) in an on-chip device can perform DNA amplification in vitro through precise thermal <b>control</b> at each step of the polymerase chain reaction (PCR)...This study reports a straightforward fabrication technique for patterning an on-chip graphene-based device with hole arrays, in which the mechanism of surface structures can achieve stable and uniform thermal <b>control</b> for the amplification of DNA fragments...The temperature <b>control</b> of the heater was performed by means of a developed programmable PCR apparatus.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/P/PL/Plant Extracts">Plant Extracts</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/30963783">

Coptidis Rhizoma: a comprehensive review of its traditional uses, botany, phytochemistry, pharmacology and toxicology.

</a>

<small>(PMID30963783</small>)

<br>...However, further research should be undertaken to investigate the clinical effects, toxic constituents, target organs and pharmacokinetics, and to establish criteria for quality <b>control</b>, for CR and its related medications.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32166607">

A Review of Coronavirus Disease-2019 (COVID-19).

</a>

<small>(PMID32166607</small>)

<br>...Prevention entails home isolation of suspected cases and those with mild illnesses and strict infection <b>control</b> measures at hospitals that include contact and droplet precautions.

<td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=mitigation+OR+containment+OR+control&from=CORD19#/H/HE/Henipavirus Infections">Henipavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31006350">

Nipah virus: epidemiology, pathology, immunobiology and advances in diagnosis, vaccine designing and control strategies - a comprehensive review.

</a>

<small>(PMID31006350</small>)

<br>....

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 4566 matches before 2019/12



"""



h = display(HTML(htmlresults))

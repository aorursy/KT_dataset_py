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
htmlprompt="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>

<h2>CORD-19 Task Details</h2>

Source: <a href="https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=563">What has been published about ethical and social science considerations?</A>



<p><strong>What has been published concerning ethical considerations for research? What has been published concerning social sciences at the outbreak response?</strong></p>





<table class=l border=1><tr><th>Kaggle prompt<th>Search terms used<th>Formatted Results

<tr><td>Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019<td>ethics OR ethical<td>Task9a results below

<tr><td>Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight<td>thematic OR oversight<td>Task9b results below

<tr><td>Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.<td>multidisciplinary OR global<td>Task9c results below

<tr><td>Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)<td>public measures<td>Task9d results below

<tr><td>Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.<td>psychological health <td>Task9e results below

</table>

"""



h = display(HTML(htmlprompt))



htmlresults="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>

<hr><a name="task9a"><b>Task9a Kaggle Prompt:</b> Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019</a><p><b>Results:</b><p>

Searching for (ethics OR ethical) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=ethics+OR+ethical&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=ethics+OR+ethical&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31806107">

Zika Vaccine Development: Current Status.

</a>

<small>(PMID31806107</small>)

<br>...In this review, we survey current vaccine efforts, preclinical and clinical results, and <b>ethical</b> and other concerns that directly bear on vaccine development.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 192 matches before 2019/12

<hr><a name="task9b"><b>Task9b Kaggle Prompt:</b> Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight</a><p><b>Results:</b><p>

Searching for (thematic OR oversight) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 65 matches before 2019/12

<hr><a name="task9c"><b>Task9c Kaggle Prompt:</b> Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.</a><p><b>Results:</b><p>

Searching for (multidisciplinary OR global) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/G/GL/Global Health">Global Health</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/T/TU/Tuberculosis">Tuberculosis</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/C/CO/Coronavirus">Coronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/I/IN/Influenza in Birds">Influenza in Birds</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/S/SP/Spike Glycoprotein, Coronavirus">Spike Glycoprotein, Coronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/G/GL/Glycoproteins">Glycoproteins</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=29><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32119825">

Feasibility of controlling COVID-19 outbreaks by isolation of cases and contacts.

</a>

<small>(PMID32119825</small>)

<br>...FUNDING: Wellcome Trust,  <b>Global</b> Challenges Research Fund, and Health Data Research UK..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32238336">

Global Telemedicine Implementation and Integration Within Health Systems to Fight the COVID-19 Pandemic: A Call to Action.

</a>

<small>(PMID32238336</small>)

<br>...Several challenges remain for the <b>global</b> use and integration of telemedicine into the public health response to COVID-19 and future outbreaks.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32231345">

Inhibition of SARS-CoV-2 (previously 2019-nCoV) infection by a highly potent pan-coronavirus fusion inhibitor targeting its spike protein that harbors a high  capacity to mediate membrane fusion.

</a>

<small>(PMID32231345</small>)

<br>...The recent outbreak of coronavirus disease (COVID-19) caused by SARS-CoV-2 infection in Wuhan, China has posed a serious threat to <b>global</b> public health.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32264957">

Fighting against the common enemy of COVID-19: a practice of building a community with a shared future for mankind.

</a>

<small>(PMID32264957</small>)

<br>...In order to prevent a potential pandemic-level outbreak of COVID-19, we, as a community of shared future for mankind, recommend for all international leaders to support preparedness in low and middle income countries  especially, take strong <b>global</b> interventions by using old approaches or new tools, mobilize global resources to equip hospital facilities and supplies to protect noisome infections and to provide personal protective tools such as facemask to general population, and quickly initiate research projects on drug and vaccine development.

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32240603">

COVID-19: global consequences for oncology.

</a>

<small>(PMID32240603</small>)

<br>....

<td>Editorial</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32155431">

Unveiling the Origin and Transmission of 2019-nCoV.

</a>

<small>(PMID32155431</small>)

<br>...A novel coronavirus has caused thousands of human infections in China since December 2019, raising a <b>global</b> public health concern.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32199470">

The global community needs to swiftly ramp up the response to contain COVID-19.

</a>

<small>(PMID32199470</small>)

<br>....

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32200634">

Protein Structure and Sequence Reanalysis of 2019-nCoV Genome Refutes Snakes as Its Intermediate Host and the Unique Similarity between Its Spike Protein Insertions and HIV-1.

</a>

<small>(PMID32200634</small>)

<br>...As the infection of 2019-nCoV coronavirus is quickly developing into a <b>global</b> pneumonia epidemic, the careful analysis of its transmission and cellular mechanisms is sorely needed.

<td>Journal Article; Research Support, N.I.H., Extramural; Research Support, U.S. Gov't, Non-P.H.S.</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31985790">

Potential for global spread of a novel coronavirus from China.

</a>

<small>(PMID31985790</small>)

<br>....

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32048560">

The global spread of 2019-nCoV: a molecular evolutionary analysis.

</a>

<small>(PMID32048560</small>)

<br>...The <b>global</b> spread of the 2019-nCoV is continuing and is fast moving, as indicated by the WHO raising the risk assessment to high.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32035997">

Persistence of coronaviruses on inanimate surfaces and their inactivation with biocidal agents.

</a>

<small>(PMID32035997</small>)

<br>...Currently, the emergence of a novel human coronavirus, SARS-CoV-2, has become a <b>global</b> health concern causing severe respiratory tract infections in humans.

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32233161">

Analysis on 54 Mortality Cases of Coronavirus Disease 2019 in the Republic of Korea from January 19 to March 10, 2020.

</a>

<small>(PMID32233161</small>)

<br>...Since the identification of the first case of coronavirus disease 2019 (COVID-19), the <b>global</b> number of confirmed cases as of March 15, 2020, is 156,400, with total death in 5,833 (3.7%) worldwide.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32182811">

Reverse Logistics Network Design for Effective Management of Medical Waste in Epidemic Outbreaks: Insights from the Coronavirus Disease 2019 (COVID-19) Outbreak in Wuhan (China).

</a>

<small>(PMID32182811</small>)

<br>...The outbreak of an epidemic disease may pose significant treats to human beings and may further lead to a <b>global</b> crisis.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32160889">

Geographical tracking and mapping of coronavirus disease COVID-19/severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) epidemic and associated events around the world: how 21st century GIS technologies are supporting the global fight against outbreaks and epidemics.

</a>

<small>(PMID32160889</small>)

<br>....

<td>Editorial</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32193904">

Drive-Through Screening Center for COVID-19: a Safe and Efficient Screening System against Massive Community Outbreak.

</a>

<small>(PMID32193904</small>)

<br>...It could be implemented in other countries to cope with the <b>global</b> COVID-19 outbreak and transformed according to their own situations..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32189081">

AI-Driven Tools for Coronavirus Outbreak: Need of Active Learning and Cross-Population Train/Test Models on Multitudinal/Multimodal Data.

</a>

<small>(PMID32189081</small>)

<br>...The novel coronavirus (COVID-19) outbreak, which was identified in late 2019, requires special attention because of its future epidemics and possible <b>global</b> threats.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32202489">

Risk of COVID-19 importation to the Pacific islands through global air travel.

</a>

<small>(PMID32202489</small>)

<br>...On 30 January 2020, WHO declared coronavirus (COVID-19) a <b>global</b> public health emergency...We analyse travel and <b>Global</b> Health Security Index data using a scoring tool to produce quantitative estimates of COVID-19 importation risk, by departing and arriving country.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32092911">

Rigidity of the Outer Shell Predicted by a Protein Intrinsic Disorder Model Sheds Light on the COVID-19 (Wuhan-2019-nCoV) Infectivity.

</a>

<small>(PMID32092911</small>)

<br>...With almost 65,000 infected, a worldwide death toll of at least 1370 (as of 14 February 2020), and with the potential to affect up to two-thirds of the world population, COVID-19 is considered by the World Health Organization (WHO) to be a <b>global</b> health emergency.

<td>Editorial</td>

<td>2020/02</td>

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

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32091395">

Estimated effectiveness of symptom and risk screening to prevent the spread of COVID-19.

</a>

<small>(PMID32091395</small>)

<br>...Traveller screening is being used to limit further spread of COVID-19 following its recent emergence, and symptom screening has become a ubiquitous tool in the <b>global</b> response.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31986257">

A novel coronavirus outbreak of global health concern.

</a>

<small>(PMID31986257</small>)

<br>....

<td>Journal Article; Comment</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31992387">

Detection of 2019 novel coronavirus (2019-nCoV) by real-time RT-PCR.

</a>

<small>(PMID31992387</small>)

<br>...Control material is made available through European Virus Archive - <b>Global</b> (EVAg), a European Union infrastructure project.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32019669">

Pattern of early human-to-human transmission of Wuhan 2019 novel coronavirus (2019-nCoV), December 2019 to January 2020.

</a>

<small>(PMID32019669</small>)

<br>...Transmission characteristics appear to be of similar magnitude to severe acute respiratory syndrome-related coronavirus (SARS-CoV) and pandemic influenza, indicating a risk of <b>global</b> spread..

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32023775">

An interim review of the epidemiological characteristics of 2019 novel coronavirus.

</a>

<small>(PMID32023775</small>)

<br>...OBJECTIVES: The 2019 novel coronavirus (2019-nCoV) from Wuhan, China is currently recognized as a public health emergency of <b>global</b> concern.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32231392">

Forecasting the novel coronavirus COVID-19.

</a>

<small>(PMID32231392</small>)

<br>...What will be the <b>global</b> impact of the novel coronavirus (COVID-19)? Answering this question requires accurate forecasting the spread of confirmed cases as well as analysis of the number of deaths and recoveries.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226284">

The global battle against SARS-CoV-2 and COVID-19.

</a>

<small>(PMID32226284</small>)

<br>....

<td>Editorial; Research Support, Non-U.S. Gov't</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226285">

SARS-CoV-2: an Emerging Coronavirus that Causes a Global Threat.

</a>

<small>(PMID32226285</small>)

<br>....

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31629079">

Identifying potential emerging threats through epidemic intelligence activities-looking for the needle in the haystack?

</a>

<small>(PMID31629079</small>)

<br>...CONCLUSIONS: PHE's manual EI process quickly and accurately detected <b>global</b> public health threats at the earliest stages and allowed for monitoring of events as they evolved..

<td>Journal Article; Systematic Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31767249">

[What are the determinants of viral outbreaks and is it possible to predict their emergence?]

</a>

<small>(PMID31767249</small>)

<br>...Finally, to guarantee that the measures taken are relevant and acceptable to the population, a <b>multidisciplinary</b> approach must be systematically relied upon and re-evaluated on a prospective basis..

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31668202">

Emerging and Reemerging Infectious Diseases: Global Overview.

</a>

<small>(PMID31668202</small>)

<br>....

<td>Editorial; Introductory Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32217506">

The Role of the Global Health Development/Eastern Mediterranean Public Health Network and the Eastern Mediterranean Field Epidemiology Training Programs in Preparedness for COVID-19.

</a>

<small>(PMID32217506</small>)

<br>...This viewpoint article aims to highlight the contribution of the <b>Global</b> Health Development (GHD)/Eastern Mediterranean Public Health Network (EMPHNET) and the EMR's Field Epidemiology Training Program (FETPs) to prepare for and respond to the current COVID-19 threat.

<td>Editorial</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32006657">

The next big threat to global health? 2019 novel coronavirus (2019-nCoV): What advice can we give to travellers? - Interim recommendations January 2020, from the Latin-American society for Travel Medicine (SLAMVI).

</a>

<small>(PMID32006657</small>)

<br>....

<td>Editorial</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/G/GL/Global Health">Global Health</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/30541664">

70 years of human rights in global health: drawing on a contentious past to secure a hopeful future.

</a>

<small>(PMID30541664</small>)

<br>....

<td>Historical Article; Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/T/TU/Tuberculosis">Tuberculosis</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31829147">

Under-reporting of TB cases and associated factors: a case study in China.

</a>

<small>(PMID31829147</small>)

<br>...BACKGROUND: Tuberculosis is a leading cause of death worldwide and has become a high <b>global</b> health priority...Having an accurate account of the number of national TB  cases is essential to understanding the national and <b>global</b> burden of the disease and in managing TB prevention and control efforts.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/C/CO/Coronavirus">Coronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32090689">

Trypsin promotes porcine deltacoronavirus mediating cell-to-cell fusion in a cell type-dependent manner.

</a>

<small>(PMID32090689</small>)

<br>...Porcine deltacoronavirus (PDCoV) is a newly emerging threat to the <b>global</b> porcine industry.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/I/IN/Influenza in Birds">Influenza in Birds</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31752591">

Role for migratory domestic poultry and/or wild birds in the global spread of avian influenza?

</a>

<small>(PMID31752591</small>)

<br>....

<td>Editorial</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/S/SP/Spike Glycoprotein, Coronavirus">Spike Glycoprotein, Coronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32178593">

Emerging WuHan (COVID-19) coronavirus: glycan shield and structure prediction of  spike glycoprotein and its interaction with human CD26.

</a>

<small>(PMID32178593</small>)

<br>...The recent outbreak of pneumonia-causing COVID-19 in China is an urgent <b>global</b> public health issue with an increase in mortality and morbidity.

<td>Letter</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/G/GL/Glycoproteins">Glycoproteins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31835168">

Therapeutic significance of beta-glucuronidase activity and its inhibitors: A review.

</a>

<small>(PMID31835168</small>)

<br>...The emergence of disease and dearth of effective pharmacological agents on most therapeutic fronts, constitutes a major threat to <b>global</b> public health and man's  existence.

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32166607">

A Review of Coronavirus Disease-2019 (COVID-19).

</a>

<small>(PMID32166607</small>)

<br>... The <b>global</b> impact of this new epidemic is yet uncertain..

<td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=multidisciplinary+OR+global&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32044389">

Going global - Travel and the 2019 novel coronavirus.

</a>

<small>(PMID32044389</small>)

<br>....

<td>Editorial</td>

<td>2020/01</td>

</tr>

</table>

<p>There are also 1691 matches before 2019/12

<hr><a name="task9d"><b>Task9d Kaggle Prompt:</b> Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)</a><p><b>Results:</b><p>

Searching for (public measures) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+measures&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+measures&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+measures&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=9><a href="http://www.softconcourse.com/CORD19/?filterText=public+measures&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32217507">

Assessment of Health Information About COVID-19 Prevention on the Internet: Infodemiological Study.

</a>

<small>(PMID32217507</small>)

<br>...CONCLUSIONS: It is necessary to urge and promote the use of the websites of official <b>public</b> health organizations when seeking information on COVID-19 preventive <b>measures</b> on  the internet.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32264957">

Fighting against the common enemy of COVID-19: a practice of building a community with a shared future for mankind.

</a>

<small>(PMID32264957</small>)

<br>...To date, we have found it is one of the greatest challenges to human beings in fighting against COVID-19 in the history, because SARS-CoV-2 is different from SARS-CoV and MERS-CoV in terms of biological features and transmissibility, and also found the containment strategies including the non-pharmaceutical <b>public</b> health <b>measures</b> implemented in China are  effective and successful.

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32088333">

Lessons learned from the 2019-nCoV epidemic on prevention of future infectious diseases.

</a>

<small>(PMID32088333</small>)

<br>...These <b>measures</b> were motivated by the need to provide effective treatment of patients, and involved consultation with three major groups in policy formulation-public health experts, the government, and the general <b>public</b>.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32052841">

Isolation, quarantine, social distancing and community containment: pivotal role  for old-style public health measures in the novel coronavirus (2019-nCoV) outbreak.

</a>

<small>(PMID32052841</small>)

<br>....

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32183901">

Epidemiology, causes, clinical manifestation and diagnosis, prevention and control of coronavirus disease (COVID-19) during the early outbreak period: a scoping review.

</a>

<small>(PMID32183901</small>)

<br>...Preventive <b>measures</b> such  as masks, hand hygiene practices, avoidance of <b>public</b> contact, case detection, contact tracing, and quarantines have been discussed as ways to reduce transmission.

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32155789">

Immediate Psychological Responses and Associated Factors during the Initial Stage of the 2019 Coronavirus Disease (COVID-19) Epidemic among the General Population  in China.

</a>

<small>(PMID32155789</small>)

<br>....

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32027631">

Initial Public Health Response and Interim Clinical Guidance for the 2019 Novel Coronavirus Outbreak - United States, December 31, 2019-February 4, 2020.

</a>

<small>(PMID32027631</small>)

<br>...Although these <b>measures</b> might not prevent the eventual establishment of ongoing, widespread transmission of the virus in the United States, they are being implemented to 1) slow the spread of illness; 2) provide time to better prepare health care systems and the general <b>public</b> to be ready if widespread transmission with substantial associated illness occurs; and 3) better characterize 2019-nCoV infection to guide public health recommendations and the development of medical countermeasures including diagnostics, therapeutics, and vaccines.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32023775">

An interim review of the epidemiological characteristics of 2019 novel coronavirus.

</a>

<small>(PMID32023775</small>)

<br>...METHODS: We reviewed the currently available literature to provide up-to-date guidance on control <b>measures</b> to be implemented by <b>public</b> health authorities...However, there remain considerable uncertainties, which should be considered when providing guidance to <b>public</b> health authorities on control <b>measures</b>.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=public+measures&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31784255">

[Epidemic and emerging prone-infectious diseases: Lessons learned and ways forward].

</a>

<small>(PMID31784255</small>)

<br>....

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=public+measures&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32166607">

A Review of Coronavirus Disease-2019 (COVID-19).

</a>

<small>(PMID32166607</small>)

<br>....

<td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

</table>

<p>There are also 327 matches before 2019/12

<hr><a name="task9e"><b>Task9e Kaggle Prompt:</b> Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.</a><p><b>Results:</b><p>

Searching for (psychological health) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=psychological+health&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=5><a href="http://www.softconcourse.com/CORD19/?filterText=psychological+health&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32155789">

Immediate Psychological Responses and Associated Factors during the Initial Stage of the 2019 Coronavirus Disease (COVID-19) Epidemic among the General Population  in China.

</a>

<small>(PMID32155789</small>)

<br>...Background: The 2019 coronavirus disease (COVID-19) epidemic is a public <b>health</b> emergency of international concern and poses a challenge to <b>psychological</b> resilience...<b>Psychological</b> impact was assessed by the Impact of Event Scale-Revised  (IES-R), and mental <b>health</b> status was assessed by the Depression, Anxiety and Stress Scale (DASS-21)...Female gender, student status, specific physical symptoms (e.g., myalgia, dizziness, coryza), and poor self-rated <b>health</b> status were significantly associated with a greater <b>psychological</b> impact of the outbreak and higher levels  of stress, anxiety, and depression (p < 0.05)...Specific up-to-date and accurate <b>health</b> information (e.g., treatment, local outbreak situation) and particular precautionary measures (e.g., hand hygiene, wearing a mask) were associated with  a lower <b>psychological</b> impact of the outbreak and lower levels of stress, anxiety, and depression (p < 0.05)...Our findings identify factors associated with a lower level of <b>psychological</b> impact and better mental <b>health</b> status that can be used to formulate psychological interventions to improve the mental health of vulnerable  groups during the COVID-19 epidemic..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32202646">

Factors Associated With Mental Health Outcomes Among Health Care Workers Exposed  to Coronavirus Disease 2019.

</a>

<small>(PMID32202646</small>)

<br>...Conclusions and Relevance: In this survey of heath  care workers in hospitals equipped with fever clinics or wards for patients with  COVID-19 in Wuhan and other regions in China, participants reported experiencing  <b>psychological</b> burden, especially nurses, women, those in Wuhan, and frontline <b>health</b> care workers directly engaged in the diagnosis, treatment, and care for patients with COVID-19..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226291">

Progression of Mental Health Services during the COVID-19 Outbreak in China.

</a>

<small>(PMID32226291</small>)

<br>...Patients, <b>health</b> professionals, and the general public are under insurmountable <b>psychological</b> pressure which may lead to various psychological problems, such as anxiety, fear, depression, and insomnia...The National <b>Health</b> Commission  of China has summoned a call for emergency <b>psychological</b> crisis intervention and  thus, various mental health associations and organizations have established expert teams to compile guidelines and public health educational articles/videos  for mental health professionals and the general public alongside with online mental health services.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226292">

Tribute to health workers in China: A group of respectable population during the  outbreak of the COVID-19.

</a>

<small>(PMID32226292</small>)

<br>....

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

</table>

<p>There are also 72 matches before 2019/12

"""



h = display(HTML(htmlresults))

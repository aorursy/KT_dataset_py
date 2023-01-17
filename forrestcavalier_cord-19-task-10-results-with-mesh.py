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

Source: <a href="https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=583">What has been published about information sharing and inter-sectoral collaboration?</A>



<p><strong>What has been published about information sharing and inter-sectoral collaboration? What has been published about data standards and nomenclature? What has been published about governmental public health? What do we know about risk communication? What has been published about communicating with high-risk populations? What has been published to clarify community measures? What has been published about equity considerations and problems of inequity?</strong></p>

<p>Specifically, we want to know what the literature reports about:</p>





<table class=l border=1><tr><th>Kaggle prompt<th>Search terms used<th>Formatted Results

<tr><td>What has been published about governmental public health?<td>public health<td>Task10a results below

<tr><td>What has been published about equity considerations and problems of inequity?<td>equity OR inequity<td>Task10b results below

<tr><td>Methods for coordinating data-gathering with standardized nomenclature.<td>data nomenclature<td>Task10c results below

<tr><td>Understanding and mitigating barriers to information-sharing.<td>information AND sharing AND barriers<td>Task10d results below

<tr><td>How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).<td>private OR non-profit OR academic<td>Task10e results below

<tr><td>Integration of federal/state/local public health surveillance systems.<td>integration AND surveillance<td>Task10f results below

<tr><td>Value of investments in baseline public health response infrastructure preparedness<td>investments OR preparedness<td>Task10g results below

<tr><td>Modes of communicating with target high-risk populations (elderly, health care workers).<td>high-risk population AND communication<td>Task10h results below

<tr><td>Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).<td>risk communication<td>Task10i results below

<tr><td>Misunderstanding around containment and mitigation.<td>misunderstanding AND containment<td>Task10j results below

<tr><td>Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.<td>mitigate AND inequality<td>Task10k results below

<tr><td>Measures to reach marginalized and disadvantaged populations.<td>marginalized OR disadvantaged OR underrepresented OR minorities<td>Task10l results below

<tr><td>Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.<td>incarcerated<td>Task10m results below

<tr><td>Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care<td>coverage policies AND barriers<td>Task10n results below

</table>



"""



h = display(HTML(htmlTaskDetails))
htmlresults="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>



<hr><a name="task10a"><b>Task10a Kaggle Prompt:</b> What has been published about governmental public health?</a><p><b>Results:</b><p>

Searching for (public health) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/R/RE/Respiratory Tract Infections">Respiratory Tract Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/V/VA/Vaccines">Vaccines</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/C/CO/Coronavirus">Coronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/L/LU/Lung">Lung</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/C/CO/Congresses as Topic">Congresses as Topic</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/G/GL/Glucuronidase">Glucuronidase</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31832255">

Influenza virus detection: driving change in public health laboratories in the Western Pacific Region.

</a>

<small>(PMID31832255</small>)

<br>....

<td>Journal Article</td>

<td>Winter 2018</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31931793">

An evaluation of the Zambia influenza sentinel surveillance system, 2011-2017.

</a>

<small>(PMID31931793</small>)

<br>...Such information would enable countries to assess the performance of their surveillance systems, identify shortfalls for improvement and provide evidence of data reliability for policy making and <b>public</b> <b>health</b> interventions.

<td>Evaluation Study; Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=10><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32119825">

Feasibility of controlling COVID-19 outbreaks by isolation of cases and contacts.

</a>

<small>(PMID32119825</small>)

<br>...We measured the success of controlling outbreaks using isolation and contact tracing, and quantified the weekly maximum number of cases traced to measure feasibility of <b>public</b> <b>health</b> effort.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32264957">

Fighting against the common enemy of COVID-19: a practice of building a community with a shared future for mankind.

</a>

<small>(PMID32264957</small>)

<br>...To date, we have found it is one of the greatest challenges to human beings in fighting against COVID-19 in the history, because SARS-CoV-2 is different from SARS-CoV and MERS-CoV in terms of biological features and transmissibility, and also found the containment strategies including the non-pharmaceutical <b>public</b> <b>health</b> measures implemented in China are  effective and successful.

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32171054">

COVID-19: a potential public health problem for homeless populations.

</a>

<small>(PMID32171054</small>)

<br>....

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32217506">

The Role of the Global Health Development/Eastern Mediterranean Public Health Network and the Eastern Mediterranean Field Epidemiology Training Programs in Preparedness for COVID-19.

</a>

<small>(PMID32217506</small>)

<br>...The World <b>Health</b> Organization (WHO) declared the current COVID-19 a <b>public</b> health emergency of international concern on January 30, 2020...Many of these countries addressed the need for increasing capacity in the areas of surveillance and rapid response to <b>public</b> <b>health</b> threats...This viewpoint article aims to highlight the contribution of the Global <b>Health</b> Development (GHD)/Eastern Mediterranean <b>Public</b> Health Network (EMPHNET) and the EMR's Field Epidemiology Training Program (FETPs) to prepare for and respond to the current COVID-19 threat...The FETPs are currently actively participating in surveillance and screening at the ports of entry, development of communication materials and guidelines, and sharing information to <b>health</b> professionals and the <b>public</b>...However, some countries remain ill-equipped, have poor diagnostic capacity, and are in need of further capacity development in response to <b>public</b> <b>health</b> threats...It is essential that GHD/EMPHNET and FETPs continue building the capacity to respond to COVID-19 and intensify support for preparedness and response to <b>public</b> <b>health</b> emergencies..

<td>Editorial</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32183901">

Epidemiology, causes, clinical manifestation and diagnosis, prevention and control of coronavirus disease (COVID-19) during the early outbreak period: a scoping review.

</a>

<small>(PMID32183901</small>)

<br>...The World <b>Health</b> Organization  has declared it a <b>Public</b> Health Emergency of International Concern...Although these studies are relevant to control the current <b>public</b> emergency, more high-quality research is needed to provide valid and reliable ways to manage this kind of public <b>health</b> emergency in both the short- and long-term..

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32053579">

Persons Evaluated for 2019 Novel Coronavirus - United States, January 2020.

</a>

<small>(PMID32053579</small>)

<br>...As of January 31, 2020, CDC had responded to clinical inquiries from <b>public</b>  <b>health</b> officials and health care providers to assist in evaluating approximately  650 persons thought to be at risk for 2019-nCoV infection.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32027631">

Initial Public Health Response and Interim Clinical Guidance for the 2019 Novel Coronavirus Outbreak - United States, December 31, 2019-February 4, 2020.

</a>

<small>(PMID32027631</small>)

<br>...On January 30, the World <b>Health</b> Organization (WHO) Director-General declared that the 2019-nCoV  outbreak constitutes a <b>Public</b> Health Emergency of International Concern.(dagger)  On January 31, the U.S...<b>public</b> <b>health</b> emergency to respond to 2019-nCoV.( section sign) Also on January 31, the president of the United States signed a "Proclamation on  Suspension of Entry as Immigrants and Nonimmigrants of Persons who Pose a Risk of Transmitting 2019 Novel Coronavirus," which limits entry into the United States of persons who traveled to mainland China to U.S...Although these measures might not prevent the eventual establishment of ongoing, widespread transmission of the virus in the United States, they are being implemented to 1) slow the spread of illness; 2) provide time to better prepare <b>health</b> care systems and the general <b>public</b> to be ready if widespread transmission with substantial associated illness occurs; and 3) better characterize 2019-nCoV infection to guide public health recommendations and the development of medical countermeasures including diagnostics, therapeutics, and vaccines...<b>Public</b> <b>health</b> authorities are monitoring the situation closely.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32236385">

Outbreak investigation in cargo ship in times of COVID-19 crisis, Port of Santos, Brazil.

</a>

<small>(PMID32236385</small>)

<br>...The investigation resulted from the implementation of the contingency plan to face a <b>public</b> <b>health</b>  emergency of international importance and several surveillance entities cooperated..

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31806107">

Zika Vaccine Development: Current Status.

</a>

<small>(PMID31806107</small>)

<br>...Zika virus outbreaks have been explosive and unpredictable and have led to significant adverse <b>health</b> effects-as well as considerable <b>public</b> anxiety.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=27><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32036774">

Emerging novel coronavirus (2019-nCoV)-current scenario, evolutionary perspective based on genome analysis and recent developments.

</a>

<small>(PMID32036774</small>)

<br>...This decade's first  CoV, named 2019-nCoV, emerged from Wuhan, China, and declared as '<b>Public</b> <b>Health</b> Emergency of International Concern' on January 30(th), 2020 by the World Health Organization (WHO).

<td>Journal Article; Review</td>

<td>2020/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32081569">

Community pharmacist in public health emergencies: Quick to action against the coronavirus 2019-nCoV outbreak.

</a>

<small>(PMID32081569</small>)

<br>...The 2019-nCoV infection that is caused by a novel strain of coronavirus was first detected in China in the end of December 2019 and declared a <b>public</b> <b>health</b> emergency of international concern by the World Health Organization on January 30, 2020.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32113880">

Corona Virus International Public Health Emergencies: Implications for Radiology  Management.

</a>

<small>(PMID32113880</small>)

<br>...This article discusses how radiology departments can most effectively respond to  this <b>public</b> <b>health</b> emergency..

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32238336">

Global Telemedicine Implementation and Integration Within Health Systems to Fight the COVID-19 Pandemic: A Call to Action.

</a>

<small>(PMID32238336</small>)

<br>...The response strategy included early diagnosis, patient isolation, symptomatic monitoring of contacts as well as suspected and confirmed cases, and <b>public</b> <b>health</b> quarantine... This framework could be applied at a large scale to improve the national <b>public</b> <b>health</b> response...Several challenges remain for the global use and integration of telemedicine into the <b>public</b> <b>health</b> response to COVID-19 and future outbreaks.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32240095">

Emergence of a Novel Coronavirus (COVID-19): Protocol for Extending Surveillance  Used by the Royal College of General Practitioners Research and Surveillance Centre and Public Health England.

</a>

<small>(PMID32240095</small>)

<br>...BACKGROUND: The Royal College of General Practitioners (RCGP) Research and Surveillance Centre (RSC) and <b>Public</b> <b>Health</b> England (PHE) have successfully worked together on the surveillance of influenza and other infectious diseases for over 50 years, including three previous pandemics.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32231345">

Inhibition of SARS-CoV-2 (previously 2019-nCoV) infection by a highly potent pan-coronavirus fusion inhibitor targeting its spike protein that harbors a high  capacity to mediate membrane fusion.

</a>

<small>(PMID32231345</small>)

<br>...The recent outbreak of coronavirus disease (COVID-19) caused by SARS-CoV-2 infection in Wuhan, China has posed a serious threat to global <b>public</b> <b>health</b>.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32217507">

Assessment of Health Information About COVID-19 Prevention on the Internet: Infodemiological Study.

</a>

<small>(PMID32217507</small>)

<br>...Most of them were produced in the United States and Spain (n=58, 73%) by digital media sources and official <b>public</b> <b>health</b> organizations (n=60, 75%)...The analysis by type of author (official <b>public</b> <b>health</b>  organizations versus digital media) revealed significant differences regarding the recommendation to wear a mask when you are healthy only if caring for a person with suspected COVID-19 (odds ratio [OR] 4.39)...CONCLUSIONS: It is necessary to urge and promote the use of the websites of official <b>public</b> <b>health</b> organizations when seeking information on COVID-19 preventive measures on  the internet.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32155431">

Unveiling the Origin and Transmission of 2019-nCoV.

</a>

<small>(PMID32155431</small>)

<br>...A novel coronavirus has caused thousands of human infections in China since December 2019, raising a global <b>public</b> <b>health</b> concern.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32087334">

The epidemic of 2019-novel-coronavirus (2019-nCoV) pneumonia and insights for emerging infectious diseases in the future.

</a>

<small>(PMID32087334</small>)

<br>...At the end of December 2019, a novel coronavirus, 2019-nCoV, caused an outbreak of pneumonia spreading from Wuhan, Hubei province, to the whole country of China, which has posed great threats to <b>public</b> <b>health</b> and attracted enormous attention around the world.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32088333">

Lessons learned from the 2019-nCoV epidemic on prevention of future infectious diseases.

</a>

<small>(PMID32088333</small>)

<br>...These measures were motivated by the need to provide effective treatment of patients, and involved consultation with three major groups in policy formulation-public <b>health</b> experts, the government, and the general <b>public</b>...This experience will provide China and other countries with valuable lessons for quickly coordinating and coping with future <b>public</b> <b>health</b> emergencies..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32132747">

Public health round-up.

</a>

<small>(PMID32132747</small>)

<br>....

<td>News</td>

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

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32015508">

A new coronavirus associated with human respiratory disease in China.

</a>

<small>(PMID32015508</small>)

<br>...Emerging infectious diseases, such as severe acute respiratory syndrome (SARS) and Zika virus disease, present a major threat to <b>public</b> <b>health</b>(1-3).

<td>Case Reports; Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32155789">

Immediate Psychological Responses and Associated Factors during the Initial Stage of the 2019 Coronavirus Disease (COVID-19) Epidemic among the General Population  in China.

</a>

<small>(PMID32155789</small>)

<br>...Background: The 2019 coronavirus disease (COVID-19) epidemic is a <b>public</b> <b>health</b> emergency of international concern and poses a challenge to psychological resilience.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32210236">

Epidemiological data from the COVID-19 outbreak, real-time case information.

</a>

<small>(PMID32210236</small>)

<br>...The generation of detailed, real-time, and robust data for emerging disease outbreaks is important and can help to generate  robust evidence that will support and inform <b>public</b> <b>health</b> decision making..

<td>Dataset; Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32202489">

Risk of COVID-19 importation to the Pacific islands through global air travel.

</a>

<small>(PMID32202489</small>)

<br>...On 30 January 2020, WHO declared coronavirus (COVID-19) a global <b>public</b> <b>health</b> emergency.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32111262">

A mathematical model for simulating the phase-based transmissibility of a novel coronavirus.

</a>

<small>(PMID32111262</small>)

<br>....

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32100667">

Passengers' destinations from China: low risk of Novel Coronavirus (2019-nCoV) transmission into Africa and South America.

</a>

<small>(PMID32100667</small>)

<br>...Increased <b>public</b> <b>health</b> response including early case recognition, isolation of identified case, contract tracing and targeted airport screening, public awareness and vigilance of health workers will help mitigate the force of  further spread to naive countries..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32046816">

Effectiveness of airport screening at detecting travellers infected with novel coronavirus (2019-nCoV).

</a>

<small>(PMID32046816</small>)

<br>...We evaluated effectiveness of thermal passenger screening for 2019-nCoV infection at airport exit and entry to inform <b>public</b> <b>health</b> decision-making.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31992387">

Detection of 2019 novel coronavirus (2019-nCoV) by real-time RT-PCR.

</a>

<small>(PMID31992387</small>)

<br>...BACKGROUND: The ongoing outbreak of the recently emerged novel coronavirus (2019-nCoV) poses a challenge for <b>public</b> <b>health</b> laboratories as virus isolates are unavailable while there is growing evidence that the outbreak is more widespread than initially thought, and international spread through travellers does already occur...AIM: We aimed to develop and deploy robust diagnostic methodology for use in <b>public</b> <b>health</b> laboratory settings without having virus material available.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32191174">

2019-nCoV: The Identify-Isolate-Inform (3I) Tool Applied to a Novel Emerging Coronavirus.

</a>

<small>(PMID32191174</small>)

<br>...Upon confirmation of a suspected 2019-nCoV case, affected persons must immediately be placed in airborne infection isolation and the appropriate <b>public</b> <b>health</b> agencies notified.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32023775">

An interim review of the epidemiological characteristics of 2019 novel coronavirus.

</a>

<small>(PMID32023775</small>)

<br>...OBJECTIVES: The 2019 novel coronavirus (2019-nCoV) from Wuhan, China is currently recognized as a <b>public</b> <b>health</b> emergency of global concern...METHODS: We reviewed the currently available literature to provide up-to-date guidance on control measures to be implemented by <b>public</b> <b>health</b> authorities...However, there remain considerable uncertainties, which should be considered when providing guidance to <b>public</b> <b>health</b> authorities on control measures.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226291">

Progression of Mental Health Services during the COVID-19 Outbreak in China.

</a>

<small>(PMID32226291</small>)

<br>...Patients, <b>health</b> professionals, and the general <b>public</b> are under insurmountable psychological pressure which may lead to various psychological problems, such as anxiety, fear, depression, and insomnia...The National <b>Health</b> Commission  of China has summoned a call for emergency psychological crisis intervention and  thus, various mental health associations and organizations have established expert teams to compile guidelines and <b>public</b> health educational articles/videos  for mental health professionals and the general public alongside with online mental health services.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32172672">

A tug-of-war between severe acute respiratory syndrome coronavirus 2 and host antiviral defence: lessons from other pathogenic viruses.

</a>

<small>(PMID32172672</small>)

<br>...World <b>Health</b> Organization has declared the ongoing outbreak of coronavirus disease 2019 (COVID-19) a <b>Public</b> Health Emergency of International Concern.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32215461">

The value of mitigating epidemic peaks of COVID-19 for more effective public health responses.

</a>

<small>(PMID32215461</small>)

<br>....

<td>Editorial</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31629079">

Identifying potential emerging threats through epidemic intelligence activities-looking for the needle in the haystack?

</a>

<small>(PMID31629079</small>)

<br>...In this study, entries captured by <b>Public</b> <b>Health</b> England's (PHE) manual event-based EI system were examined to inform future intelligence gathering activities...CONCLUSIONS: PHE's manual EI process quickly and accurately detected global <b>public</b> <b>health</b> threats at the earliest stages and allowed for monitoring of events as they evolved..

<td>Journal Article; Systematic Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31784255">

[Epidemic and emerging prone-infectious diseases: Lessons learned and ways forward].

</a>

<small>(PMID31784255</small>)

<br>...The investigation of virus detection and persistence in semen across a range of emerging viruses is useful for clinical and <b>public</b> <b>health</b> reasons, in particular for viruses that lead to high mortality or morbidity rates or to epidemics...Preparedness including management of complex humanitarian crises with community distrust is a  cornerstone in response to high consequence emerging infectious disease outbreaks and imposes strengthening of the <b>public</b> <b>health</b> response infrastructure and emergency outbreak systems in high-risk regions..

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/R/RE/Respiratory Tract Infections">Respiratory Tract Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31489496">

Paramyxoviruses respiratory syncytial virus, parainfluenza virus, and human metapneumovirus infection in pediatric hospitalized patients and climate correlation in a subtropical region of southern China: a 7-year survey.

</a>

<small>(PMID31489496</small>)

<br>...These findings will assist <b>public</b> <b>health</b> authorities and  clinicians in improving strategies for controlling paramyxovirus infection..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/V/VA/Vaccines">Vaccines</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31446443">

Advances in Vaccines.

</a>

<small>(PMID31446443</small>)

<br>...Emerging infectious diseases such as malaria, Ebola virus disease, and Zika virus disease also threaten <b>public</b> <b>health</b>  around the world.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/C/CO/Coronavirus">Coronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32209118">

Using the spike protein feature to predict infection risk and monitor the evolutionary dynamic of coronavirus.

</a>

<small>(PMID32209118</small>)

<br>...The smooth distance curve for SARS-CoV suggests that its close relatives still exist in nature and <b>public</b> <b>health</b> is challenged as usual.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/L/LU/Lung">Lung</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32004427">

First Case of 2019 Novel Coronavirus in the United States.

</a>

<small>(PMID32004427</small>)

<br>...This case highlights the importance of close coordination between clinicians and <b>public</b> <b>health</b> authorities at the local, state, and federal levels, as well as the need for rapid dissemination of clinical information related to the care of patients with this emerging infection..

<td>Case Reports; Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/C/CO/Congresses as Topic">Congresses as Topic</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32216872">

Mitigating the impact of conference and travel cancellations on researchers' futures.

</a>

<small>(PMID32216872</small>)

<br>...The need to protect <b>public</b> <b>health</b> during the current COVID-19 pandemic has necessitated conference cancellations on an unprecedented scale.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32178593">

Emerging WuHan (COVID-19) coronavirus: glycan shield and structure prediction of  spike glycoprotein and its interaction with human CD26.

</a>

<small>(PMID32178593</small>)

<br>...The recent outbreak of pneumonia-causing COVID-19 in China is an urgent global <b>public</b> <b>health</b> issue with an increase in mortality and morbidity.

<td>Letter</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32166607">

A Review of Coronavirus Disease-2019 (COVID-19).

</a>

<small>(PMID32166607</small>)

<br>...There is a new <b>public</b> <b>health</b> crises threatening the world with the emergence and  spread of 2019 novel coronavirus (2019-nCoV) or the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2).

<td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=public+health&from=CORD19#/G/GL/Glucuronidase">Glucuronidase</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31835168">

Therapeutic significance of beta-glucuronidase activity and its inhibitors: A review.

</a>

<small>(PMID31835168</small>)

<br>...The emergence of disease and dearth of effective pharmacological agents on most therapeutic fronts, constitutes a major threat to global <b>public</b> <b>health</b> and man's  existence.

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

</table>

<p>There are also 1932 matches before 2019/12

<hr><a name="task10b"><b>Task10b Kaggle Prompt:</b> What has been published about equity considerations and problems of inequity?</a><p><b>Results:</b><p>

Searching for (equity OR inequity) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 26 matches before 2019/12

<hr><a name="task10c"><b>Task10c Kaggle Prompt:</b> Methods for coordinating data-gathering with standardized nomenclature.</a><p><b>Results:</b><p>

Searching for (data nomenclature) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 10 matches before 2019/12

<hr><a name="task10d"><b>Task10d Kaggle Prompt:</b> Understanding and mitigating barriers to information-sharing.</a><p><b>Results:</b><p>

Searching for (information AND sharing) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=information+AND+sharing&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=information+AND+sharing&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32217506">

The Role of the Global Health Development/Eastern Mediterranean Public Health Network and the Eastern Mediterranean Field Epidemiology Training Programs in Preparedness for COVID-19.

</a>

<small>(PMID32217506</small>)

<br>...The FETPs are currently actively participating in surveillance and screening at the ports of entry, development of communication materials and guidelines, and <b>sharing</b> <b>information</b> to health professionals and the public.

<td>Editorial</td>

<td>2020/03</td>

</tr>

</table>

<p>There are also 39 matches before 2019/12

<hr><a name="task10e"><b>Task10e Kaggle Prompt:</b> How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).</a><p><b>Results:</b><p>

Searching for (private OR non-profit OR academic) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=private+OR+non-profit+OR+academic&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=private+OR+non-profit+OR+academic&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31992387">

Detection of 2019 novel coronavirus (2019-nCoV) by real-time RT-PCR.

</a>

<small>(PMID31992387</small>)

<br>...Through coordination between <b>academic</b> and public laboratories, we confirmed assay exclusivity based on 297 original clinical specimens containing a full spectrum of human respiratory viruses...CONCLUSION: The present study demonstrates the enormous response capacity achieved through coordination of <b>academic</b> and public laboratories in national and European research networks..

<td>Journal Article</td>

<td>2020/01</td>

</tr>

</table>

<p>There are also 252 matches before 2019/12

<hr><a name="task10f"><b>Task10f Kaggle Prompt:</b> Integration of federal/state/local public health surveillance systems.</a><p><b>Results:</b><p>

Searching for (integration AND surveillance) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 24 matches before 2019/12

<hr><a name="task10g"><b>Task10g Kaggle Prompt:</b> Value of investments in baseline public health response infrastructure preparedness</a><p><b>Results:</b><p>

Searching for (investments OR preparedness) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=investments+OR+preparedness&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=investments+OR+preparedness&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=investments+OR+preparedness&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=investments+OR+preparedness&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32264957">

Fighting against the common enemy of COVID-19: a practice of building a community with a shared future for mankind.

</a>

<small>(PMID32264957</small>)

<br>...In order to prevent a potential pandemic-level outbreak of COVID-19, we, as a community of shared future for mankind, recommend for all international leaders to support <b>preparedness</b> in low and middle income countries  especially, take strong global interventions by using old approaches or new tools, mobilize global resources to equip hospital facilities and supplies to protect noisome infections and to provide personal protective tools such as facemask to general population, and quickly initiate research projects on drug and vaccine development.

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32217506">

The Role of the Global Health Development/Eastern Mediterranean Public Health Network and the Eastern Mediterranean Field Epidemiology Training Programs in Preparedness for COVID-19.

</a>

<small>(PMID32217506</small>)

<br>...GHD/EMPHNET has the scientific expertise to contribute to elevating the level of country alert and <b>preparedness</b> in the EMR and to provide technical support through health promotion, training and training materials, guidelines, coordination, and communication...It is essential that GHD/EMPHNET and FETPs continue building the capacity to respond to COVID-19 and intensify support for <b>preparedness</b> and response to public health emergencies..

<td>Editorial</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=investments+OR+preparedness&from=CORD19#/C/CO/Communicable Diseases, Emerging">Communicable Diseases, Emerging</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31784255">

[Epidemic and emerging prone-infectious diseases: Lessons learned and ways forward].

</a>

<small>(PMID31784255</small>)

<br>...<b>Preparedness</b> including management of complex humanitarian crises with community distrust is a  cornerstone in response to high consequence emerging infectious disease outbreaks and imposes strengthening of the public health response infrastructure and emergency outbreak systems in high-risk regions..

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=4><a href="http://www.softconcourse.com/CORD19/?filterText=investments+OR+preparedness&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32081569">

Community pharmacist in public health emergencies: Quick to action against the coronavirus 2019-nCoV outbreak.

</a>

<small>(PMID32081569</small>)

<br>...Community pharmacists in one of the first areas that had confirmed cases of the viral infection, Macau, joined the collaborative force in supporting the local health emergency <b>preparedness</b> and response arrangements.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32186277">

Rapidly increasing cumulative incidence of coronavirus disease (COVID-19) in the  European Union/European Economic Area and the United Kingdom, 1 January to 15 March 2020.

</a>

<small>(PMID32186277</small>)

<br>...Based on the experience from Italy, countries, hospitals and intensive care units should increase their <b>preparedness</b> for a surge of patients with COVID-19 who will require healthcare, and in particular intensive care..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32156332">

Potential scenarios for the progression of a COVID-19 epidemic in the European Union and the European Economic Area, March 2020.

</a>

<small>(PMID32156332</small>)

<br>...We provide scenarios for use in <b>preparedness</b> for a possible widespread epidemic.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

</table>

<p>There are also 436 matches before 2019/12

<hr><a name="task10h"><b>Task10h Kaggle Prompt:</b> Modes of communicating with target high-risk populations (elderly, health care workers).</a><p><b>Results:</b><p>

Searching for (high-risk communication) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 7 matches before 2019/12

<hr><a name="task10i"><b>Task10i Kaggle Prompt:</b> Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).</a><p><b>Results:</b><p>

Searching for (risk communication) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=risk+communication&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=risk+communication&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31832249">

Learning from recent outbreaks to strengthen risk communication capacity for the  next influenza pandemic in the Western Pacific Region.

</a>

<small>(PMID31832249</small>)

<br>....

<td>Historical Article; Journal Article</td>

<td>Winter 2018</td>

</tr>

</table>

<p>There are also 96 matches before 2019/12

<hr><a name="task10j"><b>Task10j Kaggle Prompt:</b> Misunderstanding around containment and mitigation.</a><p><b>Results:</b><p>

Searching for (misunderstanding AND containment) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



<hr><a name="task10k"><b>Task10k Kaggle Prompt:</b> Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.</a><p><b>Results:</b><p>

Searching for (mitigation AND inequality) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



<hr><a name="task10l"><b>Task10l Kaggle Prompt:</b> Measures to reach marginalized and disadvantaged populations.</a><p><b>Results:</b><p>

Searching for (marginalized OR disadvantaged OR underrepresented OR minorities) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 33 matches before 2019/12

<hr><a name="task10m"><b>Task10m Kaggle Prompt:</b> Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.</a><p><b>Results:</b><p>

Searching for (prisoners) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 4 matches before 2019/12

<hr><a name="task10n"><b>Task10n Kaggle Prompt:</b> Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care</a><p><b>Results:</b><p>

Searching for (insurance coverage) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 15 matches before 2019/12



"""



h = display(HTML(htmlresults))

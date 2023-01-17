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

Source: <a href="https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=561">What do we know about vaccines and therapeutics?</A>



<p><strong>What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics?</strong></p>





<table class=l border=1><tr><th>Kaggle prompt<th>Search terms used<th>Formatted Results

<tr><td>Effectiveness of drugs being developed and tried to treat COVID-19 patients.<td>drug OR medication OR pharmaceutical OR medicine<td>Task4a results below

<tr><td>Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.<td>viral inhibitor OR naproxen OR clarithromycin OR minocyclinethat<td>Task4b results below

<tr><td>Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.<td>Antibody-Dependent Enhancement ADE<td>Task4c results below

<tr><td>Exploration of use of best animal models and their predictive value for a human vaccine.<td>animal model<td>Task4d results below

<tr><td>Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.<td>therapeutic OR antiviral OR effectiveness OR treatment<td>Task4e results below

<tr><td>Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.<td>therapeutic OR equitable OR capacity<td>Task4f results below

<tr><td>Efforts targeted at a universal coronavirus vaccine.<td>vaccine<td>Task4g results below

<tr><td>Efforts to develop animal models and standardize challenge studies<td>challenge study OR challenge OR trial<td>Task4h results below

<tr><td>Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers<td>prophylaxis OR prophylactic<td>Task4i results below

<tr><td>Approaches to evaluate risk for enhanced disease after vaccination<td>vaccine risk<td>Task4j results below

<tr><td>Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]<td>vaccine development<td>Task4k results below

</table>

"""



h = display(HTML(htmlprompt))



htmlresults="""

<style>

 .l th { text-align:left;}

  .l td { text-align:left;}

   .l tr { text-align:left;}

</style>

<hr><a name="task4a"><b>Task4a Kaggle Prompt:</b> Effectiveness of drugs being developed and tried to treat COVID-19 patients.</a><p><b>Results:</b><p>

Searching for (drug OR medication OR pharmaceutical OR medicine) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/V/VI/Viral Proteins">Viral Proteins</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/A/AN/Anti-Bacterial Agents">Anti-Bacterial Agents</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/P/PL/Plant Extracts">Plant Extracts</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/V/VA/Vaccines">Vaccines</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/E/EN/Enterovirus A, Human">Enterovirus A, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/A/AU/Autophagy">Autophagy</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/D/DO/Dogs">Dogs</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/E/EN/Endogenous Retroviruses">Endogenous Retroviruses</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32215760">

Tilorone: a Broad-Spectrum Antiviral Invented in the USA and Commercialized in Russia and beyond.

</a>

<small>(PMID32215760</small>)

<br>...This is a small-molecule orally bioavailable <b>drug</b> that was originally discovered in the USA and is currently used clinically as an  antiviral in Russia and the Ukraine...More recently we have identified additional promising antiviral activities against Middle East Respiratory Syndrome, Chikungunya, Ebola and Marburg which highlights that this old <b>drug</b> may have other uses against new viruses.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32098094">

Structural Basis for Inhibiting Porcine Epidemic Diarrhea Virus Replication with  the 3C-Like Protease Inhibitor GC376.

</a>

<small>(PMID32098094</small>)

<br>...The coronavirus 3C-like protease (3CL(pro)) has  a conserved structure and catalytic mechanism and plays a key role during viral polyprotein processing, thus serving as an appealing antiviral <b>drug</b> target.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/V/VI/Viral Proteins">Viral Proteins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32201449">

Drug targets for corona virus: A systematic review.

</a>

<small>(PMID32201449</small>)

<br>...We searched PubMed and RCSB database with keywords HCoV, NCoV, corona virus, SERS-CoV, MERS-CoV, 2019-nCoV, crystal structure, X-ray crystallography structure, NMR structure, target, and <b>drug</b> target till Feb 3, 2020...The search identified seven major targets (spike protein, envelop protein, membrane protein, protease, nucleocapsid protein, hemagglutinin esterase, and helicase) for which <b>drug</b> design can be considered...There are other 16 nonstructural proteins (NSPs),  which can also be considered from the <b>drug</b> design perspective...The major structural proteins and NSPs may serve an important role from <b>drug</b> design perspectives.

<td>Journal Article; Systematic Review</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=11><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32208840">

Clinical observation and management of COVID-19 patients.

</a>

<small>(PMID32208840</small>)

<br>...Regarding the traditional Chinese <b>medicine</b>, Professor Lu suggests to develop a creative evaluation system because of the complicated chemical compositions.

<td>Journal Article</td>

<td>2020/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32065348">

Can Chinese Medicine Be Used for Prevention of Corona Virus Disease 2019 (COVID-19)? A Review of Historical Classics, Research Evidence and Current Prevention Programs.

</a>

<small>(PMID32065348</small>)

<br>...This was followed by prevention programs recommending Chinese <b>medicine</b> (CM) for the prevention.

<td>Historical Article; Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32081569">

Community pharmacist in public health emergencies: Quick to action against the coronavirus 2019-nCoV outbreak.

</a>

<small>(PMID32081569</small>)

<br>...This paper aimed to improve the understanding of community pharmacists' role in case of 2019-CoV outbreak based on the practical experiences in consultation with the recommendations made by the International <b>Pharmaceutical</b> Federation on the Coronavirus 2019-nCoV outbreak..

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32264957">

Fighting against the common enemy of COVID-19: a practice of building a community with a shared future for mankind.

</a>

<small>(PMID32264957</small>)

<br>...In order to prevent a potential pandemic-level outbreak of COVID-19, we, as a community of shared future for mankind, recommend for all international leaders to support preparedness in low and middle income countries  especially, take strong global interventions by using old approaches or new tools, mobilize global resources to equip hospital facilities and supplies to protect noisome infections and to provide personal protective tools such as facemask to general population, and quickly initiate research projects on <b>drug</b> and vaccine development.

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32209231">

Computers and viral diseases. Preliminary bioinformatics studies on the design of a synthetic vaccine and a preventative peptidomimetic antagonist against the SARS-CoV-2 (2019-nCoV, COVID-19) coronavirus.

</a>

<small>(PMID32209231</small>)

<br>...The project was originally directed towards a use case for the Q-UEL language and its implementation in a knowledge management and automated inference system for <b>medicine</b> called the BioIngine, but focus here remains mostly on the virus itself.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32087334">

The epidemic of 2019-novel-coronavirus (2019-nCoV) pneumonia and insights for emerging infectious diseases in the future.

</a>

<small>(PMID32087334</small>)

<br>...Intensive research on the novel emerging human infectious coronaviruses is urgently needed to elucidate their route of transmission and pathogenic mechanisms, and to identify potential  <b>drug</b> targets, which would promote the development of effective preventive and therapeutic countermeasures.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32113846">

In silico screening of Chinese herbal medicines with the potential to directly inhibit 2019 novel coronavirus.

</a>

<small>(PMID32113846</small>)

<br>...Resulting compounds were cross-checked for listing in the Traditional Chinese <b>Medicine</b> Systems Pharmacology Database.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32234064">

Covid-19 in China: ten critical issues for intensive care medicine.

</a>

<small>(PMID32234064</small>)

<br>....

<td>Editorial</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32117569">

Therapeutic strategies in an outbreak scenario to treat the novel coronavirus originating in Wuhan, China.

</a>

<small>(PMID32117569</small>)

<br>... I consider the options of <b>drug</b> repurposing, developing neutralizing monoclonal antibody therapy, and an oligonucleotide strategy targeting the viral RNA genome, emphasizing the promise and pitfalls of these approaches...The sequence of the ACE2-Fc protein is provided to investigators, allowing its possible use in recombinant protein expression systems to start producing <b>drug</b> today to treat patients under compassionate use, while formal clinical trials are later undertaken.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226288">

Traditional Chinese Medicine in the Treatment of Patients Infected with 2019-New  Coronavirus (SARS-CoV-2): A Review and Perspective.

</a>

<small>(PMID32226288</small>)

<br>...At the top of these conventional therapies, greater than  85% of SARS-CoV-2 infected patients in China are receiving Traditional Chinese <b>Medicine</b> (TCM) treatment.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/A/AN/Anti-Bacterial Agents">Anti-Bacterial Agents</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31548072">

Effect of antibiotic treatment in preweaned Holstein calves after experimental bacterial challenge with Pasteurella multocida.

</a>

<small>(PMID31548072</small>)

<br>...Holstein bull calves (n = 39) were transported to the University of Wisconsin-Madison School of Veterinary <b>Medicine</b> isolation facility at the mean (+/-SD) age of 52 +/- 6 d.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/P/PL/Plant Extracts">Plant Extracts</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/30963783">

Coptidis Rhizoma: a comprehensive review of its traditional uses, botany, phytochemistry, pharmacology and toxicology.

</a>

<small>(PMID30963783</small>)

<br>...dissertations; the state and local <b>drug</b> standards; PubMed; CNKI; Scopus; the Web of Science; and Google Scholar using the keywords Coptis, Coptidis Rhizoma, Huanglian, and goldthread...CONCLUSIONS:  As an important herbal <b>medicine</b> in Chinese medicine, CR has the potential to treat various diseases.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32006657">

The next big threat to global health? 2019 novel coronavirus (2019-nCoV): What advice can we give to travellers? - Interim recommendations January 2020, from the Latin-American society for Travel Medicine (SLAMVI).

</a>

<small>(PMID32006657</small>)

<br>....

<td>Editorial</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/V/VA/Vaccines">Vaccines</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31446443">

Advances in Vaccines.

</a>

<small>(PMID31446443</small>)

<br>...Vaccines represent one of the most important advances in science and <b>medicine</b>, helping people around the world in preventing the spread of infectious diseases.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/E/EN/Enterovirus A, Human">Enterovirus A, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31863580">

A nucleobase-binding pocket in a viral RNA-dependent RNA polymerase contributes to elongation complex stability.

</a>

<small>(PMID31863580</small>)

<br>...Potential applications in antiviral <b>drug</b> and vaccine development are also discussed..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/A/AU/Autophagy">Autophagy</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31433241">

Long non-coding RNA MEG3 attends to morphine-mediated autophagy of HT22 cells through modulating ERK pathway.

</a>

<small>(PMID31433241</small>)

<br>...Some long non-coding RNAs (lncRNAs) have been proposed to engage in <b>drug</b> addiction...More experiments are also needed in the future to analyse the influence of other lncRNAs in <b>drug</b> addiction..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/D/DO/Dogs">Dogs</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31541785">

Triplex doppler ultrasonography to describe the uterine arteries during diestrus  and progesterone profile in pregnant and non-pregnant bitches of different sizes.

</a>

<small>(PMID31541785</small>)

<br>...Hemodynamics of uterine vascularization is modified throughout pregnancy to meet  the increasing demand of the growing fetuses and triplex doppler ultrasonography  is widely used in human <b>medicine</b> to study the uterine arteries and assess the fetal and placental conditions.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=drug+OR+medication+OR+pharmaceutical+OR+medicine&from=CORD19#/E/EN/Endogenous Retroviruses">Endogenous Retroviruses</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31813465">

Evolutionary Medicine of Retroviruses in the Human Genome.

</a>

<small>(PMID31813465</small>)

<br>...Finally, we present perspectives of virology in the field  of evolutionary <b>medicine</b>..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 2113 matches before 2019/12

<hr><a name="task4b"><b>Task4b Kaggle Prompt:</b> Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.</a><p><b>Results:</b><p>

Searching for (viral inhibitor OR naproxen OR clarithromycin OR minocyclinethat) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=viral+inhibitor+OR+naproxen+OR+clarithromycin+OR+minocyclinethat&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=viral+inhibitor+OR+naproxen+OR+clarithromycin+OR+minocyclinethat&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=viral+inhibitor+OR+naproxen+OR+clarithromycin+OR+minocyclinethat&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32098094">

Structural Basis for Inhibiting Porcine Epidemic Diarrhea Virus Replication with  the 3C-Like Protease Inhibitor GC376.

</a>

<small>(PMID32098094</small>)

<br>....

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32040667">

Critical care management of adults with community-acquired severe respiratory viral infection.

</a>

<small>(PMID32040667</small>)

<br>....

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=5><a href="http://www.softconcourse.com/CORD19/?filterText=viral+inhibitor+OR+naproxen+OR+clarithromycin+OR+minocyclinethat&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32231345">

Inhibition of SARS-CoV-2 (previously 2019-nCoV) infection by a highly potent pan-coronavirus fusion inhibitor targeting its spike protein that harbors a high  capacity to mediate membrane fusion.

</a>

<small>(PMID32231345</small>)

<br>....

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31834525">

miR-142a-3p promotes the proliferation of porcine hemagglutinating encephalomyelitis virus by targeting Rab3a.

</a>

<small>(PMID31834525</small>)

<br>...Downregulation of miR-142a-3p by an miRNA <b>inhibitor</b> led to a significant repression of <b>viral</b> proliferation, implying that it acts as a positive regulator of PHEV proliferation...Conversely, the use of an miR-142a-3p <b>inhibitor</b> or overexpression of Rab3a resulted in a marked restriction of <b>viral</b> production at both the mRNA and protein level.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32090691">

Swine acute diarrhea syndrome coronavirus-induced apoptosis is caspase- and cyclophilin D- dependent.

</a>

<small>(PMID32090691</small>)

<br>...Moreover,  Vero E6 and IPI-2I cells treated with cyclosporin A (CsA), an <b>inhibitor</b> of mitochondrial permeability transition pore (MPTP) opening, were completely protected from SADS-CoV-induced apoptosis and <b>viral</b> replication, suggesting the involvement of cyclophilin D (CypD) in these processes.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31864417">

Antiviral activity of interleukin-11 as a response to porcine epidemic diarrhea virus infection.

</a>

<small>(PMID31864417</small>)

<br>....

<td>Journal Article</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 492 matches before 2019/12

<hr><a name="task4c"><b>Task4c Kaggle Prompt:</b> Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.</a><p><b>Results:</b><p>

Searching for (Antibody-Dependent Enhancement ADE) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>



</table>

<p>There are also 33 matches before 2019/12

<hr><a name="task4d"><b>Task4d Kaggle Prompt:</b> Exploration of use of best animal models and their predictive value for a human vaccine.</a><p><b>Results:</b><p>

Searching for (animal model) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=animal+model&from=CORD19#/C/CA/Camelids, New World">Camelids, New World</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=animal+model&from=CORD19#/C/CI/Circovirus">Circovirus</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=animal+model&from=CORD19#/C/CA/Camelids, New World">Camelids, New World</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31733447">

Deiminated proteins in extracellular vesicles and serum of llama (Lama glama)-Novel insights into camelid immunity.

</a>

<small>(PMID31733447</small>)

<br>...1758) as a <b>model</b> <b>animal</b>.

<td>Journal Article; Research Support, Non-U.S. Gov't; Research Support, U.S. Gov't, Non-P.H.S.</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=animal+model&from=CORD19#/C/CI/Circovirus">Circovirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31753544">

UV-C irradiation is able to inactivate pathogens found in commercially collected  porcine plasma as demonstrated by swine bioassay.

</a>

<small>(PMID31753544</small>)

<br>....

<td>Journal Article</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 577 matches before 2019/12

<hr><a name="task4e"><b>Task4e Kaggle Prompt:</b> Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.</a><p><b>Results:</b><p>

Searching for (therapeutic OR antiviral OR effectiveness OR treatment) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/S/SA/SARS Virus">SARS Virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/S/SE/Severe Acute Respiratory Syndrome">Severe Acute Respiratory Syndrome</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/A/AN/Antibodies, Monoclonal">Antibodies, Monoclonal</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CA/Cattle Diseases">Cattle Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/M/ME/Membrane Proteins">Membrane Proteins</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/I/IN/Interferon Type I">Interferon Type I</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/M/MA/Macrophages">Macrophages</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/M/MU/Murine hepatitis virus">Murine hepatitis virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CO/Coronavirus">Coronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/P/PO/Porcine respiratory and reproductive syndrome virus">Porcine respiratory and reproductive syndrome virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/P/PR/Proteins">Proteins</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/I/IN/Inflammation">Inflammation</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/T/TU/Tuberculosis">Tuberculosis</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/M/MI/MicroRNAs">MicroRNAs</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/R/RH/Rhinovirus">Rhinovirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/G/GL/Glycoproteins">Glycoproteins</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/E/EN/Enterovirus A, Human">Enterovirus A, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/Z/ZI/Zika Virus">Zika Virus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/H/HO/Horse Diseases">Horse Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CA/Cardiovascular Diseases">Cardiovascular Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/H/HE/Henipavirus Infections">Henipavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/M/MI/Middle East Respiratory Syndrome Coronavirus">Middle East Respiratory Syndrome Coronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CH/Cholera">Cholera</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CR/Croup">Croup</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/B/BI/Bird Diseases">Bird Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/L/LE/Lectins, C-Type">Lectins, C-Type</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/L/LI/Lipocalin-2">Lipocalin-2</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/A/AZ/Azetidines">Azetidines</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=11><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31724441">

Inhibition of SARS-CoV 3CL protease by flavonoids.

</a>

<small>(PMID31724441</small>)

<br>...The <b>antiviral</b> activity of some flavonoids against CoVs is presumed directly caused by inhibiting 3C-like protease (3CLpro).

<td>Journal Article</td>

<td>2020/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32119961">

Anti-HCV, nucleotide inhibitors, repurposing against COVID-19.

</a>

<small>(PMID32119961</small>)

<br>...KEY FINDINGS: The results suggest the <b>effectiveness</b> of Sofosbuvir, IDX-184, Ribavirin, and Remidisvir as potent drugs against the newly emerged HCoV disease.

<td>Journal Article</td>

<td>2020/05</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32054787">

Prophylactic and therapeutic remdesivir (GS-5734) treatment in the rhesus macaque model of MERS-CoV infection.

</a>

<small>(PMID32054787</small>)

<br>...The continued emergence of Middle East Respiratory Syndrome (MERS) cases with a high case fatality rate stresses the need for the availability of effective <b>antiviral</b> treatments...Here, we tested the efficacy of prophylactic and <b>therapeutic</b> remdesivir treatment in a nonhuman primate model of  MERS-CoV infection, the rhesus macaque...Prophylactic remdesivir <b>treatment</b> initiated 24 h prior to inoculation completely prevented MERS-CoV-induced clinical disease, strongly inhibited MERS-CoV replication in respiratory tissues, and prevented the formation of lung lesions...<b>Therapeutic</b> remdesivir treatment initiated 12 h postinoculation also provided a clear clinical benefit, with a reduction in clinical signs, reduced virus replication in the lungs, and decreased presence and severity of lung lesions...The data presented here support  testing of the efficacy of remdesivir <b>treatment</b> in the context of a MERS clinical trial.

<td>Journal Article; Research Support, N.I.H., Intramural</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32215760">

Tilorone: a Broad-Spectrum Antiviral Invented in the USA and Commercialized in Russia and beyond.

</a>

<small>(PMID32215760</small>)

<br>...This is a small-molecule orally bioavailable drug that was originally discovered in the USA and is currently used clinically as an  <b>antiviral</b> in Russia and the Ukraine...Over the years there have been numerous clinical and non-clinical reports of its broad spectrum of <b>antiviral</b> activity...More recently we have identified additional promising <b>antiviral</b> activities against Middle East Respiratory Syndrome, Chikungunya, Ebola and Marburg which highlights that this old drug may have other uses against new viruses.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32098094">

Structural Basis for Inhibiting Porcine Epidemic Diarrhea Virus Replication with  the 3C-Like Protease Inhibitor GC376.

</a>

<small>(PMID32098094</small>)

<br>...The coronavirus 3C-like protease (3CL(pro)) has  a conserved structure and catalytic mechanism and plays a key role during viral polyprotein processing, thus serving as an appealing <b>antiviral</b> drug target...This study helps us to understand better the PEDV 3CL(pro) substrate specificity, providing information on the optimization of GC376 for development as an antiviral <b>therapeutic</b> against coronaviruses..

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32040667">

Critical care management of adults with community-acquired severe respiratory viral infection.

</a>

<small>(PMID32040667</small>)

<br>...Oseltamivir is the most widely used neuraminidase inhibitor for <b>treatment</b> of influenza; data suggest that early use is associated with reduced mortality in critically ill patients with influenza...At present, there are no <b>antiviral</b> therapies of proven efficacy for other severe RVIs.

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31841728">

Identification of 6'-beta-fluoro-homoaristeromycin as a potent inhibitor of chikungunya virus replication.

</a>

<small>(PMID31841728</small>)

<br>...Among  the compounds tested, 6'-beta-fluoro-homoaristeromycin 3a showed potent <b>antiviral</b> activity (EC50 = 0.12 muM) against the CHIKV, without noticeable cytotoxicity up  to 250 muM...Only 3a displayed anti-CHIKV activity, whereas both3a and 3b inhibited SAH hydrolase with similar IC50 values (0.36 and 0.37 muM, respectively), which suggested that 3a's <b>antiviral</b> activity did not merely depend on the inhibition of SAH hydrolase...This is further supported by the fact that the <b>antiviral</b> effect was specific for CHIKV and some other alphaviruses and none  of the homologated analogues inhibited other RNA viruses, such as SARS-CoV, MERS-CoV, and ZIKV.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31914458">

Preclinical evaluation of AT-527, a novel guanosine nucleotide prodrug with potent, pan-genotypic activity against hepatitis C virus.

</a>

<small>(PMID31914458</small>)

<br>...Despite the availability of highly effective direct-acting <b>antiviral</b> (DAA) regimens for the treatment of hepatitis C virus (HCV) infections, sustained viral response (SVR) rates remain suboptimal for difficult-to-treat patient populations such as those with HCV genotype 3, cirrhosis or prior treatment experience, warranting development of more potent HCV replication antivirals...These favorable preclinical attributes support the ongoing clinical development of AT-527 and suggest that, when used in combination with an HCV DAA from a different class, AT-527 may increase SVR rates, especially for difficult-to-treat patient populations, and could potentially shorten <b>treatment</b> duration for all patients..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31531682">

Saikosaponin C exerts anti-HBV effects by attenuating HNF1alpha and HNF4alpha expression to suppress HBV pgRNA synthesis.

</a>

<small>(PMID31531682</small>)

<br>...These results indicate that SSc acts as a promising compound for modulating pgRNA transcription in the <b>therapeutic</b> strategies against HBV infection..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31586832">

Antitumor and antiviral activities of 4-substituted 1,2,3-triazolyl-2,3-dibenzyl-L-ascorbic acid derivatives.

</a>

<small>(PMID31586832</small>)

<br>...Two series of 6-(1,2,3-triazolyl)-2,3-dibenzyl-l-ascorbic acid derivatives with the hydroxyethylene (8a-8u) and ethylidene linkers (10c-10p) were synthesized and evaluated for their antiproliferative activity against seven malignant tumor cell lines and <b>antiviral</b> activity against a broad range of viruses.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=40><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32208840">

Clinical observation and management of COVID-19 patients.

</a>

<small>(PMID32208840</small>)

<br>...For severe or critically ill patients, in addition to the respiratory supportive <b>treatment</b>, timely multiorgan evaluation and treatment is very crucial.

<td>Journal Article</td>

<td>2020/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32100486">

2019 Novel Coronavirus (COVID-19) Pneumonia: Serial Computed Tomography Findings.

</a>

<small>(PMID32100486</small>)

<br>...After <b>treatment</b>, the lesions were shown to be almost absorbed leaving the fibrous lesions..

<td>Case Reports</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32109443">

Clinical and computed tomographic imaging features of novel coronavirus pneumonia caused by SARS-CoV-2.

</a>

<small>(PMID32109443</small>)

<br>...MATERIALS AND METHODS: A retrospective analysis was performed on the imaging findings of patients confirmed with COVID-19 pneumonia who had chest CT scanning  and <b>treatment</b> after disease onset...CT scanning provides important bases for early diagnosis and <b>treatment</b>  of NCP..

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32065348">

Can Chinese Medicine Be Used for Prevention of Corona Virus Disease 2019 (COVID-19)? A Review of Historical Classics, Research Evidence and Current Prevention Programs.

</a>

<small>(PMID32065348</small>)

<br>...METHODS: Historical records on prevention and <b>treatment</b> of infections in CM classics, clinical evidence of CM on the prevention of severe acute respiratory syndrome (SARS) and H1N1 influenza, and CM prevention programs issued by health authorities in China since the COVID-19 outbreak were retrieved from different databases and websites till 12 February, 2020.

<td>Historical Article; Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32119825">

Feasibility of controlling COVID-19 outbreaks by isolation of cases and contacts.

</a>

<small>(PMID32119825</small>)

<br>...We used the model to quantify the  potential <b>effectiveness</b> of contact tracing and isolation of cases at controlling  a severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2)-like pathogen.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32112886">

Characteristics of COVID-19 infection in Beijing.

</a>

<small>(PMID32112886</small>)

<br>...FINDINGS: By Feb 10, 2020, 262 patients  were transferred from the hospitals across Beijing to the designated hospitals for special <b>treatment</b> of the COVID-19 infected by Beijing emergency medical service.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32240095">

Emergence of a Novel Coronavirus (COVID-19): Protocol for Extending Surveillance  Used by the Royal College of General Practitioners Research and Surveillance Centre and Public Health England.

</a>

<small>(PMID32240095</small>)

<br>...At the same time and separately, the RCGP RSC's surveillance has been extended to monitor the temporal and geographical distribution of COVID-19 infection in the community as well as assess the <b>effectiveness</b> of the containment strategy...OBJECTIVES: The aims of this study are to surveil COVID-19 in both asymptomatic populations and ambulatory cases with respiratory infections, ascertain both the rate and pattern of COVID-19 spread, and assess the <b>effectiveness</b> of the containment policy...CONCLUSIONS: We have rapidly converted the established national RCGP RSC influenza surveillance system into one that can test the <b>effectiveness</b> of the COVID-19 containment policy.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32231345">

Inhibition of SARS-CoV-2 (previously 2019-nCoV) infection by a highly potent pan-coronavirus fusion inhibitor targeting its spike protein that harbors a high  capacity to mediate membrane fusion.

</a>

<small>(PMID32231345</small>)

<br>...Intranasal application of EK1C4 before or after challenge with HCoV-OC43 protected mice from infection, suggesting that EK1C4 could be used for prevention and <b>treatment</b> of infection by the currently circulating SARS-CoV-2 and other emerging SARSr-CoVs..

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32241301">

A novel treatment approach to the novel coronavirus: an argument for the use of therapeutic plasma exchange for fulminant COVID-19.

</a>

<small>(PMID32241301</small>)

<br>....

<td>Editorial</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32209231">

Computers and viral diseases. Preliminary bioinformatics studies on the design of a synthetic vaccine and a preventative peptidomimetic antagonist against the SARS-CoV-2 (2019-nCoV, COVID-19) coronavirus.

</a>

<small>(PMID32209231</small>)

<br>...This is to find a short section or sections of viral protein sequence suitable for preliminary design proposal for a peptide synthetic vaccine and a peptidomimetic  <b>therapeutic</b>, and to explore some design possibilities.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32087334">

The epidemic of 2019-novel-coronavirus (2019-nCoV) pneumonia and insights for emerging infectious diseases in the future.

</a>

<small>(PMID32087334</small>)

<br>...To date, there are no clinically approved vaccines or <b>antiviral</b> drugs available for these human coronavirus infections...Intensive research on the novel emerging human infectious coronaviruses is urgently needed to elucidate their route of transmission and pathogenic mechanisms, and to identify potential  drug targets, which would promote the development of effective preventive and <b>therapeutic</b> countermeasures...Herein, we describe the epidemic and etiological characteristics of 2019-nCoV, discuss its essential biological features, including tropism and receptor usage, summarize approaches for disease prevention and <b>treatment</b>, and speculate on the transmission route of 2019-nCoV..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32088333">

Lessons learned from the 2019-nCoV epidemic on prevention of future infectious diseases.

</a>

<small>(PMID32088333</small>)

<br>...These preliminary data suggest the <b>effectiveness</b> of a traffic restriction policy for this pandemic thus far...These measures were motivated by the need to provide effective <b>treatment</b> of patients, and involved consultation with three major groups in policy formulation-public health experts, the government, and the general public.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32048163">

Clinical and biochemical indexes from 2019-nCoV infected patients linked to viral loads and lung injury.

</a>

<small>(PMID32048163</small>)

<br>...Our results suggest a number of potential diagnosis biomarkers and angiotensin receptor blocker (ARB) drugs for potential repurposing <b>treatment</b> of 2019-nCoV infection..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32178711">

Treatment of COVID-19: old tricks for new challenges.

</a>

<small>(PMID32178711</small>)

<br>....

<td>Editorial</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32182811">

Reverse Logistics Network Design for Effective Management of Medical Waste in Epidemic Outbreaks: Insights from the Coronavirus Disease 2019 (COVID-19) Outbreak in Wuhan (China).

</a>

<small>(PMID32182811</small>)

<br>...Due to the limitation on available data and knowledge at present stage, more real-world information are needed to assess the <b>effectiveness</b> of the current solution..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32183901">

Epidemiology, causes, clinical manifestation and diagnosis, prevention and control of coronavirus disease (COVID-19) during the early outbreak period: a scoping review.

</a>

<small>(PMID32183901</small>)

<br>...To date, no specific <b>antiviral</b> treatment has proven effective; hence, infected people primarily rely on symptomatic treatment and supportive care.

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32155789">

Immediate Psychological Responses and Associated Factors during the Initial Stage of the 2019 Coronavirus Disease (COVID-19) Epidemic among the General Population  in China.

</a>

<small>(PMID32155789</small>)

<br>...Specific up-to-date and accurate health information (e.g., <b>treatment</b>, local outbreak situation) and particular precautionary measures (e.g., hand hygiene, wearing a mask) were associated with  a lower psychological impact of the outbreak and lower levels of stress, anxiety, and depression (p < 0.05).

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32216653">

Managing Oncology Services During a Major Coronavirus Outbreak: Lessons From the  Saudi Arabia Experience.

</a>

<small>(PMID32216653</small>)

<br>...Outbreaks of infectious etiology, particularly those caused by a novel virus that has no known <b>treatment</b> or vaccine, may result in the interruption of medical care provided to patients with cancer and put them at risk for undertreatment in addition to the risk of being exposed to infection, a life-threatening event among patients with cancer...This article describes the approach used to manage patients with cancer during a large-scale Middle East respiratory syndrome-coronavirus hospital outbreak in Saudi Arabia to ensure continuity of care and minimize harm from <b>treatment</b> interruption or acquiring infection.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32193905">

First Pediatric Case of Coronavirus Disease 2019 in Korea.

</a>

<small>(PMID32193905</small>)

<br>...In this  report, we present mild clinical course of her pneumonia that did not require <b>antiviral</b> treatment and serial viral test results from multiple specimens.

<td>Case Reports; Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32202646">

Factors Associated With Mental Health Outcomes Among Health Care Workers Exposed  to Coronavirus Disease 2019.

</a>

<small>(PMID32202646</small>)

<br>...Frontline health care workers engaged in direct diagnosis, <b>treatment</b>, and care of patients  with COVID-19 were associated with a higher risk of symptoms of depression (OR, 1.52; 95% CI, 1.11-2.09; P = .01), anxiety (OR, 1.57; 95% CI, 1.22-2.02; P < .001), insomnia (OR, 2.97; 95% CI, 1.92-4.60; P < .001), and distress (OR, 1.60;  95% CI, 1.25-2.04; P < .001)...Conclusions and Relevance: In this survey of heath  care workers in hospitals equipped with fever clinics or wards for patients with  COVID-19 in Wuhan and other regions in China, participants reported experiencing  psychological burden, especially nurses, women, those in Wuhan, and frontline health care workers directly engaged in the diagnosis, <b>treatment</b>, and care for patients with COVID-19..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32091395">

Estimated effectiveness of symptom and risk screening to prevent the spread of COVID-19.

</a>

<small>(PMID32091395</small>)

<br>...Previously, we developed a mathematical model to understand factors governing the <b>effectiveness</b> of traveller screening to prevent spread of emerging pathogens (Gostic et al., 2015).

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32046816">

Effectiveness of airport screening at detecting travellers infected with novel coronavirus (2019-nCoV).

</a>

<small>(PMID32046816</small>)

<br>...We evaluated <b>effectiveness</b> of thermal passenger screening for 2019-nCoV infection at airport exit and entry to inform public health decision-making.

<td>Journal Article</td>

<td>2020/02</td>

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

<br>...Here, we show that remdesivir (RDV) and IFNb have superior <b>antiviral</b> activity to LPV and RTV in vitro...In mice, both prophylactic and <b>therapeutic</b> RDV improve pulmonary function and reduce lung viral loads and severe lung pathology...<b>Therapeutic</b> LPV/RTV-IFNb improves pulmonary function but does not reduce virus replication or severe lung  pathology.

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31670218">

Porcine deltacoronavirus (PDCoV) modulates calcium influx to favor viral replication.

</a>

<small>(PMID31670218</small>)

<br>...<b>Treatment</b> with Ca(2+) channel blockers, particularly the L-type Ca(2+) channel blocker diltiazem hydrochloride, inhibited PDCoV infection significantly.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32117569">

Therapeutic strategies in an outbreak scenario to treat the novel coronavirus originating in Wuhan, China.

</a>

<small>(PMID32117569</small>)

<br>...Finally, I advocate for the fastest strategy to develop a <b>treatment</b> now, which could be resistant to any  mutations the virus may have in the future...Such a <b>treatment</b> could help infected patients before a protective vaccine is developed and widely available in the coming months to year(s)..

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32005086">

An emerging coronavirus causing pneumonia outbreak in Wuhan, China: calling for developing therapeutic and prophylactic strategies.

</a>

<small>(PMID32005086</small>)

<br>....

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31987001">

Genomic characterization of the 2019 novel human-pathogenic coronavirus isolated  from a patient with atypical pneumonia after visiting Wuhan.

</a>

<small>(PMID31987001</small>)

<br>...These findings provide the basis for starting further studies on the pathogenesis, and optimizing the design of diagnostic, <b>antiviral</b> and vaccination strategies for this emerging infection..

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226295">

COVID-19: what has been learned and to be learned about the novel coronavirus disease.

</a>

<small>(PMID32226295</small>)

<br>...We will cover the basics about the epidemiology, etiology, virology, diagnosis, <b>treatment</b>, prognosis, and prevention of the disease.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226288">

Traditional Chinese Medicine in the Treatment of Patients Infected with 2019-New  Coronavirus (SARS-CoV-2): A Review and Perspective.

</a>

<small>(PMID32226288</small>)

<br>...No specific anti-virus drugs or vaccines  are available for the <b>treatment</b> of this sudden and lethal disease...The supportive care and non-specific <b>treatment</b> to ameliorate the symptoms of the patient are the only options currently...At the top of these conventional therapies, greater than  85% of SARS-CoV-2 infected patients in China are receiving Traditional Chinese Medicine (TCM) <b>treatment</b>...In this article, relevant published literatures are thoroughly reviewed and current applications of TCM in the <b>treatment</b> of COVID-19  patients are analyzed...Due to the homology in epidemiology, genomics, and pathogenesis of the SARS-CoV-2 and SARS-CoV, and the widely use of TCM in the <b>treatment</b> of SARS-CoV, the clinical evidence showing the beneficial effect of TCM in the treatment of patients with SARS coronaviral infections are discussed...Current experiment studies that provide an insight into the mechanism underlying  the <b>therapeutic</b> effect of TCM, and those studies identified novel naturally occurring compounds with anti-coronaviral activity are also introduced..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226289">

Perspectives on therapeutic neutralizing antibodies against the Novel Coronavirus SARS-CoV-2.

</a>

<small>(PMID32226289</small>)

<br>...Current efforts are focusing on development of specific <b>antiviral</b> drugs...<b>Therapeutic</b> neutralizing antibodies (NAbs) against SARS-CoV-2 will be greatly important therapeutic agents for the treatment of coronavirus disease 2019 (COVID-19)...Although many challenges exist, NAbs still offer a <b>therapeutic</b> option to control the current pandemic and the possible re-emergence of the virus in the future, and their development therefore remains a high priority..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226290">

Targeting the Endocytic Pathway and Autophagy Process as a Novel Therapeutic Strategy in COVID-19.

</a>

<small>(PMID32226290</small>)

<br>...As a result, the endocytic pathway including endosome and lysosome has become important targets for development of <b>therapeutic</b> strategies in combating diseases caused by CoVs...In this mini-review, we will focus on the importance of the endocytic pathway as well as the autophagy process in viral infection of several pathogenic CoVs inclusive of SARS-CoV, MERS-CoV and the new  CoV named as severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), and discuss the development of <b>therapeutic</b> agents by targeting these processes...Such  knowledge will provide important clues for control of the ongoing epidemic of SARS-CoV-2 infection and <b>treatment</b> of COVID-19..

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32172669">

Diagnosis and clinical management of severe acute respiratory syndrome Coronavirus 2 (SARS-CoV-2) infection: an operational recommendation of Peking Union Medical College Hospital (V2.0).

</a>

<small>(PMID32172669</small>)

<br>...To standardize the diagnosis and <b>treatment</b> of  this new infectious disease, an operational recommendation for the diagnosis and  management of SARS-CoV-2 infection is developed by Peking Union Medical College Hospital..

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32172672">

A tug-of-war between severe acute respiratory syndrome coronavirus 2 and host antiviral defence: lessons from other pathogenic viruses.

</a>

<small>(PMID32172672</small>)

<br>...Here, we review the discovery, zoonotic origin, animal hosts, transmissibility and pathogenicity of SARS-CoV-2 in relation to its interplay with host <b>antiviral</b> defense...Important questions concerning the interaction between SARS-CoV-2 and host <b>antiviral</b> defence, including asymptomatic and presymptomatic virus shedding, are  also discussed..

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31668197">

The Middle East Respiratory Syndrome (MERS).

</a>

<small>(PMID31668197</small>)

<br>...There is no specific <b>treatment</b>.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31852899">

SKP2 attenuates autophagy through Beclin1-ubiquitination and its inhibition reduces MERS-Coronavirus infection.

</a>

<small>(PMID31852899</small>)

<br>...The SKP2-BECN1 link constitutes a promising target for host-directed <b>antiviral</b> drugs and possibly other autophagy-sensitive conditions..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31864417">

Antiviral activity of interleukin-11 as a response to porcine epidemic diarrhea virus infection.

</a>

<small>(PMID31864417</small>)

<br>...The potential of IL-11 to be used as  a novel <b>therapeutic</b> against devastating viral diarrhea in piglets deserves more attention and study..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31463771">

Short hairpin RNAs targeting M and N genes reduce replication of porcine deltacoronavirus in ST cells.

</a>

<small>(PMID31463771</small>)

<br>...This is believed to be the first report to show that shRNAs targeting the M and N genes of PDCoV exert <b>antiviral</b> effects in vitro, which suggests that RNAi is a promising new strategy against PDCoV infection..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31629079">

Identifying potential emerging threats through epidemic intelligence activities-looking for the needle in the haystack?

</a>

<small>(PMID31629079</small>)

<br>...The top five diseases in terms of the number of entries were described in depth to determine the <b>effectiveness</b> of PHE's EI surveillance system compared to other sources.

<td>Journal Article; Systematic Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31989546">

Epidemic Influenza Seasons from 2008 to 2018 in Poland: A Focused Review of Virological Characteristics.

</a>

<small>(PMID31989546</small>)

<br>...The number of confirmations, enabling a prompt commencement of <b>antiviral</b> treatment, related to the number of specimens collected from patients and on the virological situation in a given season.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/S/SA/SARS Virus">SARS Virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31668196">

Severe Acute Respiratory Syndrome: Historical, Epidemiologic, and Clinical Features.

</a>

<small>(PMID31668196</small>)

<br>...<b>Treatment</b> involves supportive care...There are no specific <b>antiviral</b> treatments or vaccines available..

<td>Historical Article; Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/S/SE/Severe Acute Respiratory Syndrome">Severe Acute Respiratory Syndrome</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32125455">

Angiotensin-converting enzyme 2 (ACE2) as a SARS-CoV-2 receptor: molecular mechanisms and potential therapeutic target.

</a>

<small>(PMID32125455</small>)

<br>....

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/A/AN/Antibodies, Monoclonal">Antibodies, Monoclonal</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32065055">

Potent binding of 2019 novel coronavirus spike protein by a SARS coronavirus-specific human monoclonal antibody.

</a>

<small>(PMID32065055</small>)

<br>...Currently, however, there is no specific <b>antiviral</b> treatment or vaccine...Considering the relatively high identity of receptor-binding domain (RBD) in 2019-nCoV and SARS-CoV, it is urgent to assess the cross-reactivity of anti-SARS CoV antibodies with 2019-nCoV spike protein, which could have important implications for rapid development of vaccines and <b>therapeutic</b> antibodies against 2019-nCoV...These results suggest that CR3022 may have the potential to be developed as candidate therapeutics, alone or in combination with other neutralizing antibodies, for the prevention and <b>treatment</b>  of 2019-nCoV infections.

<td>Letter</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CA/Cattle Diseases">Cattle Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31548072">

Effect of antibiotic treatment in preweaned Holstein calves after experimental bacterial challenge with Pasteurella multocida.

</a>

<small>(PMID31548072</small>)

<br>...Calves were randomized to receive ampicillin [n = 17, <b>treatment</b> (TX), 6.6 mg/kg i.m...At  the time of first <b>treatment</b>, consolidation had developed in 28/30 calves (TX, n = 17; CON, n = 11) and 6% (1 out of 17) of TX and 9% (1 out of 11) of CON calves had a positive respiratory score...<b>Treatment</b> failures might be due to incomplete resolution of the initial lung infection.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31606222">

Comparison of oral, intravenous, and subcutaneous fluid therapy for resuscitation of calves with diarrhea.

</a>

<small>(PMID31606222</small>)

<br>...Neonatal diarrhea remains the primary cause of mortality in dairy calves around the world, and optimal <b>treatment</b> protocols are needed...<b>Treatment</b> began when calves had severe diarrhea and had a decrease in plasma volume of at least 10%...Calves were randomly assigned to 1 of 4 <b>treatment</b> groups  of 8 to 9 calves per group: (1) OES; (2) OES with hypertonic saline (4 mL/kg, IV); (3) IV fluids (lactated Ringer's, 2 L); or (4) SC fluids (lactated Ringer's, 2 L)...Changes in plasma volume, blood pH, electrolyte levels, and physical examination scores were determined before therapy and again at 1, 2, 4, 8, and 12 h after each <b>treatment</b>...Subcutaneous fluids by themselves are a poor <b>treatment</b> option and should be only be used as supportive therapy following  the initial correction of hypovolemia and metabolic acidosis..

<td>Comparative Study; Journal Article; Randomized Controlled Trial, Veterinary</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/M/ME/Membrane Proteins">Membrane Proteins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31513894">

Production of anti-Trichophyton rubrum egg yolk immunoglobulin and its therapeutic potential for treating dermatophytosis.

</a>

<small>(PMID31513894</small>)

<br>...The aim of this study was to estimate the <b>therapeutic</b> potential of specific egg yolk immunoglobulin (IgY) on dermatophytosis caused by Trichophyton rubrum...rubrum in vitro and <b>therapeutic</b> effect on T...rubrum in vitro and a significant dose-dependent <b>therapeutic</b> effect on T.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/I/IN/Interferon Type I">Interferon Type I</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31784108">

Type I Interferon Signaling Disrupts the Hepatic Urea Cycle and Alters Systemic Metabolism to Suppress T Cell Function.

</a>

<small>(PMID31784108</small>)

<br>...Infections induce complex host responses linked to <b>antiviral</b> defense, inflammation, and tissue damage and repair.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/M/MA/Macrophages">Macrophages</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31710162">

Tumor-associated macrophages secrete CC-chemokine ligand 2 and induce tamoxifen resistance by activating PI3K/Akt/mTOR in breast cancer.

</a>

<small>(PMID31710162</small>)

<br>...We conclude that CCL2 secreted by TAM activates PI3K/Akt/mTOR signaling and promotes an endocrine resistance feedback loop in the TME, suggesting that CCL2 and TAM may be novel <b>therapeutic</b> targets for patients with endocrine-resistant breast cancer..

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31825972">

IL-4/IL-13 polarization of macrophages enhances Ebola virus glycoprotein-dependent infection.

</a>

<small>(PMID31825972</small>)

<br>...Macrophages polarized towards a M2-like anti-inflammatory state by combined IL-4  and IL-13 <b>treatment</b> were more susceptible to rVSV/EBOV GP, but not to wild-type VSV (rVSV/G), suggesting that EBOV GP-dependent entry events were enhanced by these cytokines.

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/M/MU/Murine hepatitis virus">Murine hepatitis virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32150576">

Structure of mouse coronavirus spike protein complexed with receptor reveals mechanism for viral entry.

</a>

<small>(PMID32150576</small>)

<br>...Using protease sensitivity and negative-stain EM analyses, we further showed that after protease <b>treatment</b> of the spike, receptor binding facilitated the dissociation of S1 from S2, allowing S2 to transition from pre-fusion to post-fusion conformation.

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CO/Coronavirus">Coronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32090689">

Trypsin promotes porcine deltacoronavirus mediating cell-to-cell fusion in a cell type-dependent manner.

</a>

<small>(PMID32090689</small>)

<br>...This knowledge can potentially contribute to improvement of virus production efficiency in culture, not only for vaccine preparation but also to develop <b>antiviral</b> treatments..

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/P/PO/Porcine respiratory and reproductive syndrome virus">Porcine respiratory and reproductive syndrome virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31767072">

S100A9 regulates porcine reproductive and respiratory syndrome virus replication  by interacting with the viral nucleocapsid protein.

</a>

<small>(PMID31767072</small>)

<br>...Moreover, we also found that the mutant S100A9 (E78Q) protein exhibited decreased <b>antiviral</b> activity against PRRSV compared with the parent S100A9.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/P/PR/Proteins">Proteins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31477883">

Lysosomes as a therapeutic target.

</a>

<small>(PMID31477883</small>)

<br>....

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/I/IN/Inflammation">Inflammation</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32007299">

Equine Inflammatory Markers in the Twenty-First Century: A Focus on Serum Amyloid A.

</a>

<small>(PMID32007299</small>)

<br>...The practitioner is encouraged to use SAA in conjunction with physical examination and other diagnostic modalities to guide <b>treatment</b> and monitor case progression..

<td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/T/TU/Tuberculosis">Tuberculosis</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31829147">

Under-reporting of TB cases and associated factors: a case study in China.

</a>

<small>(PMID31829147</small>)

<br>...China, which has the third largest TB epidemic in the world and has developed a reporting system to help with the control and prevention of TB, this study examined its <b>effectiveness</b> in Eastern China.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/M/MI/MicroRNAs">MicroRNAs</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31925708">

From Endothelium to Lipids, Through microRNAs and PCSK9: A Fascinating Travel Across Atherosclerosis.

</a>

<small>(PMID31925708</small>)

<br>...Recently, proprotein convertase subtilisin/kexin type 9 (PCSK9) has  been recognized as a fundamental regulator of LDL-C and anti-PCSK9 monoclonal antibodies have been approved for <b>therapeutic</b> use in hypercholesterolemia, with the promise to subvert the natural history of the disease...Identification of pivotal keystone molecules bridging lipid metabolism, endothelial dysfunction and atherogenesis will provide the mechanistic substrate to test valuable targets for prediction, prevention and <b>treatment</b> of atherosclerosis-related disease..

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/R/RH/Rhinovirus">Rhinovirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31513433">

Antiviral immunity is impaired in COPD patients with frequent exacerbations.

</a>

<small>(PMID31513433</small>)

<br>...Patients with frequent exacerbations represent a chronic obstructive pulmonary disease (COPD) subgroup requiring better <b>treatment</b> options...Frequent exacerbators had reduced sputum cell mRNA expression of the <b>antiviral</b> immune mediators type I and III interferons and reduced interferon-stimulated gene (ISG) expression when clinically stable and during virus-associated exacerbation...<b>Therapeutic</b> approaches to boost innate antimicrobial immunity in the lung could be a viable strategy for prevention and treatment of frequent exacerbations..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/G/GL/Glycoproteins">Glycoproteins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31835168">

Therapeutic significance of beta-glucuronidase activity and its inhibitors: A review.

</a>

<small>(PMID31835168</small>)

<br>...The emergence of disease and dearth of effective pharmacological agents on most <b>therapeutic</b> fronts, constitutes a major threat to global public health and man's  existence...To this end, accumulating empirical evidence supports molecular target therapy as a plausible egress and, beta-glucuronidase (betaGLU) - a lysosomal acid hydrolase responsible for the catalytic deconjugation of beta-d-glucuronides has emerged as a viable molecular target for several <b>therapeutic</b> applications...The aim is to proffer a platform on which new scaffolds can be modelled for improved betaGLU inhibitory potency and the development of new <b>therapeutic</b> agents in consequential..

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32166607">

A Review of Coronavirus Disease-2019 (COVID-19).

</a>

<small>(PMID32166607</small>)

<br>...Treatment is essentially supportive; role of <b>antiviral</b> agents is yet to be established.

<td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/E/EN/Enterovirus A, Human">Enterovirus A, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31863580">

A nucleobase-binding pocket in a viral RNA-dependent RNA polymerase contributes to elongation complex stability.

</a>

<small>(PMID31863580</small>)

<br>...The enterovirus 71 (EV71) 3Dpol is an RNA-dependent RNA polymerase (RdRP) that plays the central role in the viral genome replication, and is an important target in <b>antiviral</b> studies...Potential applications in <b>antiviral</b> drug and vaccine development are also discussed..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/Z/ZI/Zika Virus">Zika Virus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31884173">

A new multiplex RT-qPCR method for the simultaneous detection and discrimination  of Zika and chikungunya viruses.

</a>

<small>(PMID31884173</small>)

<br>...Such multiplex methods enable early and efficient diagnosis, leading to rapid <b>treatment</b> and effective confinement in outbreak cases.

<td>Journal Article; Validation Study</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/H/HO/Horse Diseases">Horse Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31982231">

Clinical Pathology in the Adult Sick Horse: The Gastrointestinal System and Liver.

</a>

<small>(PMID31982231</small>)

<br>...Hematologic and biochemical analysis can be helpful for identifying organ dysfunction, narrowing down the differential diagnostic list, and monitoring progress and response to <b>treatment</b>.

<td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31587977">

Gastrointestinal Disorders of Donkeys and Mules.

</a>

<small>(PMID31587977</small>)

<br>...Diagnosis, management, and <b>treatment</b> of conditions affecting the gastrointestinal tract from stomach to rectum, including liver and pancreas, are  discussed..

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CA/Cardiovascular Diseases">Cardiovascular Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31427727">

Counter-regulatory renin-angiotensin system in cardiovascular disease.

</a>

<small>(PMID31427727</small>)

<br>...This counter-regulatory renin-angiotensin system has a central role in the pathogenesis and development of various cardiovascular diseases and, therefore, represents a potential <b>therapeutic</b> target...In this Review, we provide the latest insights into the complexity and interplay of the components of the non-canonical renin-angiotensin system, and discuss the function and <b>therapeutic</b> potential of targeting this system to treat  cardiovascular disease..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/H/HE/Henipavirus Infections">Henipavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31006350">

Nipah virus: epidemiology, pathology, immunobiology and advances in diagnosis, vaccine designing and control strategies - a comprehensive review.

</a>

<small>(PMID31006350</small>)

<br>...High pathogenicity of NiV in humans, and lack of vaccines or therapeutics to counter this disease have attracted attention of researchers worldwide for developing effective NiV vaccine and <b>treatment</b> regimens..

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/M/MI/Middle East Respiratory Syndrome Coronavirus">Middle East Respiratory Syndrome Coronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31614169">

Preparation of virus-like particle mimetic nanovesicles displaying the S protein  of Middle East respiratory syndrome coronavirus using insect cells.

</a>

<small>(PMID31614169</small>)

<br>...By surfactant <b>treatment</b> and mechanical extrusion using S protein- or three structural protein-expressing Bm5 cells, S protein-displaying nanovesicles with diameters of approximately 100-200nm were prepared and confirmed by immuno-TEM.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CH/Cholera">Cholera</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31905206">

Risk perception and behavioral change during epidemics: Comparing models of individual and collective learning.

</a>

<small>(PMID31905206</small>)

<br>...It requires a deep understanding of how individuals perceive risks and communicate about the <b>effectiveness</b> of protective  measures, highlighting learning and social interaction as the core mechanisms driving such processes.

<td>Comparative Study; Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/C/CR/Croup">Croup</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31542653">

Defining atypical croup: A case report and review of the literature.

</a>

<small>(PMID31542653</small>)

<br>...OBJECTIVES: This case report and systematic review aims to synthesize the published literature on the definition, diagnosis and <b>treatment</b> of atypical croup.

<td>Case Reports; Journal Article; Systematic Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/B/BI/Bird Diseases">Bird Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31759453">

Clinical Management of Avian Renal Disease.

</a>

<small>(PMID31759453</small>)

<br>...<b>Treatment</b> of avian renal disease relies on supportive care, such as fluid therapy and nutritional support.

<td>Journal Article; Review</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/L/LE/Lectins, C-Type">Lectins, C-Type</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32152943">

CLEC5A: A Promiscuous Pattern Recognition Receptor to Microbes and Beyond.

</a>

<small>(PMID32152943</small>)

<br>...Thus, CLEC5A is a promiscuous pattern recognition receptor in myeloid cells and is a potential <b>therapeutic</b> target for attenuation of both septic and aseptic inflammatory reactions..

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/L/LI/Lipocalin-2">Lipocalin-2</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31901947">

The role of lipocalin-2 in age-related macular degeneration (AMD).

</a>

<small>(PMID31901947</small>)

<br>...We elaborate on the signaling cascades which trigger LCN-2 upregulation in AMD and suggest <b>therapeutic</b> strategies for targeting such pathways..

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+antiviral+OR+effectiveness+OR+treatment&from=CORD19#/A/AZ/Azetidines">Azetidines</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32032529">

Baricitinib as potential treatment for 2019-nCoV acute respiratory disease.

</a>

<small>(PMID32032529</small>)

<br>....

<td>Letter</td>

<td>2020/02</td>

</tr>

</table>

<p>There are also 7283 matches before 2019/12

<hr><a name="task4f"><b>Task4f Kaggle Prompt:</b> Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.</a><p><b>Results:</b><p>

Searching for (therapeutic OR equitable OR capacity) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/V/VI/Virus Diseases">Virus Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/A/AN/Antibodies, Monoclonal">Antibodies, Monoclonal</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/S/SE/Severe Acute Respiratory Syndrome">Severe Acute Respiratory Syndrome</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/P/PR/Proteins">Proteins</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/M/MA/Macrophages">Macrophages</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/M/ME/Membrane Proteins">Membrane Proteins</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/C/CA/Cardiovascular Diseases">Cardiovascular Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/M/MI/MicroRNAs">MicroRNAs</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/R/RH/Rhinovirus">Rhinovirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/G/GL/Glycoproteins">Glycoproteins</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/R/RE/Receptors, Pattern Recognition">Receptors, Pattern Recognition</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/L/LI/Lipocalin-2">Lipocalin-2</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/C/CO/Congresses as Topic">Congresses as Topic</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=4><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32054787">

Prophylactic and therapeutic remdesivir (GS-5734) treatment in the rhesus macaque model of MERS-CoV infection.

</a>

<small>(PMID32054787</small>)

<br>...Here, we tested the efficacy of prophylactic and <b>therapeutic</b> remdesivir treatment in a nonhuman primate model of  MERS-CoV infection, the rhesus macaque...<b>Therapeutic</b> remdesivir treatment initiated 12 h postinoculation also provided a clear clinical benefit, with a reduction in clinical signs, reduced virus replication in the lungs, and decreased presence and severity of lung lesions.

<td>Journal Article; Research Support, N.I.H., Intramural</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32098094">

Structural Basis for Inhibiting Porcine Epidemic Diarrhea Virus Replication with  the 3C-Like Protease Inhibitor GC376.

</a>

<small>(PMID32098094</small>)

<br>...This study helps us to understand better the PEDV 3CL(pro) substrate specificity, providing information on the optimization of GC376 for development as an antiviral <b>therapeutic</b> against coronaviruses..

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31531682">

Saikosaponin C exerts anti-HBV effects by attenuating HNF1alpha and HNF4alpha expression to suppress HBV pgRNA synthesis.

</a>

<small>(PMID31531682</small>)

<br>...These results indicate that SSc acts as a promising compound for modulating pgRNA transcription in the <b>therapeutic</b> strategies against HBV infection..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=16><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32231345">

Inhibition of SARS-CoV-2 (previously 2019-nCoV) infection by a highly potent pan-coronavirus fusion inhibitor targeting its spike protein that harbors a high  capacity to mediate membrane fusion.

</a>

<small>(PMID32231345</small>)

<br>...Therefore, we herein established a SARS-CoV-2 spike (S) protein-mediated cell-cell fusion assay and found that SARS-CoV-2 showed a superior plasma membrane fusion <b>capacity</b> compared to that of SARS-CoV.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32217507">

Assessment of Health Information About COVID-19 Prevention on the Internet: Infodemiological Study.

</a>

<small>(PMID32217507</small>)

<br>...BACKGROUND: The internet is a large source of health information and has the <b>capacity</b> to influence its users.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32241301">

A novel treatment approach to the novel coronavirus: an argument for the use of therapeutic plasma exchange for fulminant COVID-19.

</a>

<small>(PMID32241301</small>)

<br>....

<td>Editorial</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32209231">

Computers and viral diseases. Preliminary bioinformatics studies on the design of a synthetic vaccine and a preventative peptidomimetic antagonist against the SARS-CoV-2 (2019-nCoV, COVID-19) coronavirus.

</a>

<small>(PMID32209231</small>)

<br>...This is to find a short section or sections of viral protein sequence suitable for preliminary design proposal for a peptide synthetic vaccine and a peptidomimetic  <b>therapeutic</b>, and to explore some design possibilities.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32087334">

The epidemic of 2019-novel-coronavirus (2019-nCoV) pneumonia and insights for emerging infectious diseases in the future.

</a>

<small>(PMID32087334</small>)

<br>...Intensive research on the novel emerging human infectious coronaviruses is urgently needed to elucidate their route of transmission and pathogenic mechanisms, and to identify potential  drug targets, which would promote the development of effective preventive and <b>therapeutic</b> countermeasures.

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32193904">

Drive-Through Screening Center for COVID-19: a Safe and Efficient Screening System against Massive Community Outbreak.

</a>

<small>(PMID32193904</small>)

<br>...Increased testing <b>capacity</b> over 100 tests per day and prevention of cross-infection between testees in the waiting space are the major  advantages, while protection of staff from the outdoor atmosphere is challenging.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32046815">

Laboratory readiness and response for novel coronavirus (2019-nCoV) in expert laboratories in 30 EU/EEA countries, January 2020.

</a>

<small>(PMID32046815</small>)

<br>...We assessed the required expertise and <b>capacity</b> for molecular detection of 2019-nCoV in specialised laboratories in 30 European Union/European Economic Area (EU/EEA) countries.

<td>Journal Article</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31992387">

Detection of 2019 novel coronavirus (2019-nCoV) by real-time RT-PCR.

</a>

<small>(PMID31992387</small>)

<br>...CONCLUSION: The present study demonstrates the enormous response <b>capacity</b> achieved through coordination of academic and public laboratories in national and European research networks..

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31924756">

Comparative therapeutic efficacy of remdesivir and combination lopinavir, ritonavir, and interferon beta against MERS-CoV.

</a>

<small>(PMID31924756</small>)

<br>...In mice, both prophylactic and <b>therapeutic</b> RDV improve pulmonary function and reduce lung viral loads and severe lung pathology...<b>Therapeutic</b> LPV/RTV-IFNb improves pulmonary function but does not reduce virus replication or severe lung  pathology.

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32117569">

Therapeutic strategies in an outbreak scenario to treat the novel coronavirus originating in Wuhan, China.

</a>

<small>(PMID32117569</small>)

<br>....

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32005086">

An emerging coronavirus causing pneumonia outbreak in Wuhan, China: calling for developing therapeutic and prophylactic strategies.

</a>

<small>(PMID32005086</small>)

<br>....

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226288">

Traditional Chinese Medicine in the Treatment of Patients Infected with 2019-New  Coronavirus (SARS-CoV-2): A Review and Perspective.

</a>

<small>(PMID32226288</small>)

<br>...Current experiment studies that provide an insight into the mechanism underlying  the <b>therapeutic</b> effect of TCM, and those studies identified novel naturally occurring compounds with anti-coronaviral activity are also introduced..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226289">

Perspectives on therapeutic neutralizing antibodies against the Novel Coronavirus SARS-CoV-2.

</a>

<small>(PMID32226289</small>)

<br>...<b>Therapeutic</b> neutralizing antibodies (NAbs) against SARS-CoV-2 will be greatly important therapeutic agents for the treatment of coronavirus disease 2019 (COVID-19)...Although many challenges exist, NAbs still offer a <b>therapeutic</b> option to control the current pandemic and the possible re-emergence of the virus in the future, and their development therefore remains a high priority..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226290">

Targeting the Endocytic Pathway and Autophagy Process as a Novel Therapeutic Strategy in COVID-19.

</a>

<small>(PMID32226290</small>)

<br>...As a result, the endocytic pathway including endosome and lysosome has become important targets for development of <b>therapeutic</b> strategies in combating diseases caused by CoVs...In this mini-review, we will focus on the importance of the endocytic pathway as well as the autophagy process in viral infection of several pathogenic CoVs inclusive of SARS-CoV, MERS-CoV and the new  CoV named as severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), and discuss the development of <b>therapeutic</b> agents by targeting these processes.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31864417">

Antiviral activity of interleukin-11 as a response to porcine epidemic diarrhea virus infection.

</a>

<small>(PMID31864417</small>)

<br>...The potential of IL-11 to be used as  a novel <b>therapeutic</b> against devastating viral diarrhea in piglets deserves more attention and study..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31832249">

Learning from recent outbreaks to strengthen risk communication capacity for the  next influenza pandemic in the Western Pacific Region.

</a>

<small>(PMID31832249</small>)

<br>....

<td>Historical Article; Journal Article</td>

<td>Winter 2018</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/V/VI/Virus Diseases">Virus Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31790139">

Pathological consequences of the unfolded protein response and downstream protein disulphide isomerases in pulmonary viral infection and disease.

</a>

<small>(PMID31790139</small>)

<br>...Protein folding within the endoplasmic reticulum (ER) exists in a delicate balance; perturbations of this balance can overload the folding <b>capacity</b> of the ER and disruptions of ER homoeostasis is implicated in numerous diseases.

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/A/AN/Antibodies, Monoclonal">Antibodies, Monoclonal</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32065055">

Potent binding of 2019 novel coronavirus spike protein by a SARS coronavirus-specific human monoclonal antibody.

</a>

<small>(PMID32065055</small>)

<br>...Considering the relatively high identity of receptor-binding domain (RBD) in 2019-nCoV and SARS-CoV, it is urgent to assess the cross-reactivity of anti-SARS CoV antibodies with 2019-nCoV spike protein, which could have important implications for rapid development of vaccines and <b>therapeutic</b> antibodies against 2019-nCoV.

<td>Letter</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31526954">

Detection of MERS-CoV antigen on formalin-fixed paraffin-embedded nasal tissue of alpacas by immunohistochemistry using human monoclonal antibodies directed against different epitopes of the spike protein.

</a>

<small>(PMID31526954</small>)

<br>...In summary, three tested human mAbs demonstrated <b>capacity</b> for detection of MERS-CoV antigen on FFPE samples and may be implemented in double or triple immunohistochemical methods..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32217506">

The Role of the Global Health Development/Eastern Mediterranean Public Health Network and the Eastern Mediterranean Field Epidemiology Training Programs in Preparedness for COVID-19.

</a>

<small>(PMID32217506</small>)

<br>...Countries in the Eastern  Mediterranean Region (EMR) have a high vulnerability and variable <b>capacity</b> to respond to outbreaks...Many of these countries addressed the need for increasing <b>capacity</b> in the areas of surveillance and rapid response to public health threats...However, some countries remain ill-equipped, have poor diagnostic <b>capacity</b>, and are in need of further capacity development in response to public health threats...It is essential that GHD/EMPHNET and FETPs continue building the <b>capacity</b> to respond to COVID-19 and intensify support for preparedness and response to public health emergencies..

<td>Editorial</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/S/SE/Severe Acute Respiratory Syndrome">Severe Acute Respiratory Syndrome</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32125455">

Angiotensin-converting enzyme 2 (ACE2) as a SARS-CoV-2 receptor: molecular mechanisms and potential therapeutic target.

</a>

<small>(PMID32125455</small>)

<br>....

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/P/PR/Proteins">Proteins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31477883">

Lysosomes as a therapeutic target.

</a>

<small>(PMID31477883</small>)

<br>....

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/M/MA/Macrophages">Macrophages</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31710162">

Tumor-associated macrophages secrete CC-chemokine ligand 2 and induce tamoxifen resistance by activating PI3K/Akt/mTOR in breast cancer.

</a>

<small>(PMID31710162</small>)

<br>...We conclude that CCL2 secreted by TAM activates PI3K/Akt/mTOR signaling and promotes an endocrine resistance feedback loop in the TME, suggesting that CCL2 and TAM may be novel <b>therapeutic</b> targets for patients with endocrine-resistant breast cancer..

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/M/ME/Membrane Proteins">Membrane Proteins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31513894">

Production of anti-Trichophyton rubrum egg yolk immunoglobulin and its therapeutic potential for treating dermatophytosis.

</a>

<small>(PMID31513894</small>)

<br>...The aim of this study was to estimate the <b>therapeutic</b> potential of specific egg yolk immunoglobulin (IgY) on dermatophytosis caused by Trichophyton rubrum...rubrum in vitro and <b>therapeutic</b> effect on T...rubrum in vitro and a significant dose-dependent <b>therapeutic</b> effect on T.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/C/CA/Cardiovascular Diseases">Cardiovascular Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31427727">

Counter-regulatory renin-angiotensin system in cardiovascular disease.

</a>

<small>(PMID31427727</small>)

<br>...This counter-regulatory renin-angiotensin system has a central role in the pathogenesis and development of various cardiovascular diseases and, therefore, represents a potential <b>therapeutic</b> target...In this Review, we provide the latest insights into the complexity and interplay of the components of the non-canonical renin-angiotensin system, and discuss the function and <b>therapeutic</b> potential of targeting this system to treat  cardiovascular disease..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/M/MI/MicroRNAs">MicroRNAs</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31925708">

From Endothelium to Lipids, Through microRNAs and PCSK9: A Fascinating Travel Across Atherosclerosis.

</a>

<small>(PMID31925708</small>)

<br>...Recently, proprotein convertase subtilisin/kexin type 9 (PCSK9) has  been recognized as a fundamental regulator of LDL-C and anti-PCSK9 monoclonal antibodies have been approved for <b>therapeutic</b> use in hypercholesterolemia, with the promise to subvert the natural history of the disease.

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/R/RH/Rhinovirus">Rhinovirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31513433">

Antiviral immunity is impaired in COPD patients with frequent exacerbations.

</a>

<small>(PMID31513433</small>)

<br>...<b>Therapeutic</b> approaches to boost innate antimicrobial immunity in the lung could be a viable strategy for prevention and treatment of frequent exacerbations..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/G/GL/Glycoproteins">Glycoproteins</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31835168">

Therapeutic significance of beta-glucuronidase activity and its inhibitors: A review.

</a>

<small>(PMID31835168</small>)

<br>...The emergence of disease and dearth of effective pharmacological agents on most <b>therapeutic</b> fronts, constitutes a major threat to global public health and man's  existence...To this end, accumulating empirical evidence supports molecular target therapy as a plausible egress and, beta-glucuronidase (betaGLU) - a lysosomal acid hydrolase responsible for the catalytic deconjugation of beta-d-glucuronides has emerged as a viable molecular target for several <b>therapeutic</b> applications...The aim is to proffer a platform on which new scaffolds can be modelled for improved betaGLU inhibitory potency and the development of new <b>therapeutic</b> agents in consequential..

<td>Journal Article; Review</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/R/RE/Receptors, Pattern Recognition">Receptors, Pattern Recognition</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32152943">

CLEC5A: A Promiscuous Pattern Recognition Receptor to Microbes and Beyond.

</a>

<small>(PMID32152943</small>)

<br>...Thus, CLEC5A is a promiscuous pattern recognition receptor in myeloid cells and is a potential <b>therapeutic</b> target for attenuation of both septic and aseptic inflammatory reactions..

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/L/LI/Lipocalin-2">Lipocalin-2</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31901947">

The role of lipocalin-2 in age-related macular degeneration (AMD).

</a>

<small>(PMID31901947</small>)

<br>...We elaborate on the signaling cascades which trigger LCN-2 upregulation in AMD and suggest <b>therapeutic</b> strategies for targeting such pathways..

<td>Journal Article; Review</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=therapeutic+OR+equitable+OR+capacity&from=CORD19#/C/CO/Congresses as Topic">Congresses as Topic</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32216872">

Mitigating the impact of conference and travel cancellations on researchers' futures.

</a>

<small>(PMID32216872</small>)

<br>...The proposed solutions may also offer long-term benefits for those who normally cannot attend conferences, and thus lead to a more <b>equitable</b> future  for generations of researchers..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

</table>

<p>There are also 2725 matches before 2019/12

<hr><a name="task4g"><b>Task4g Kaggle Prompt:</b> Efforts targeted at a universal coronavirus vaccine.</a><p><b>Results:</b><p>

Searching for (vaccine) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/V/VI/Viral Vaccines">Viral Vaccines</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/A/AN/Antibodies, Viral">Antibodies, Viral</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/V/VA/Vaccines">Vaccines</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/R/RE/Respiratory Syncytial Virus Infections">Respiratory Syncytial Virus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/C/CO/Coronavirus">Coronavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/V/VA/Vaccines, Virus-Like Particle">Vaccines, Virus-Like Particle</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/P/PL/Plants">Plants</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/H/HE/Henipavirus Infections">Henipavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/E/EN/Enterovirus A, Human">Enterovirus A, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/E/EB/Ebolavirus">Ebolavirus</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/V/VI/Viral Vaccines">Viral Vaccines</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31950289">

Recombinant adenovirus carrying a core neutralizing epitope of porcine epidemic diarrhea virus and heat-labile enterotoxin B of Escherichia coli as a mucosal vaccine.

</a>

<small>(PMID31950289</small>)

<br>...To  protect against PEDV invasion, a mucosal <b>vaccine</b> is utilized effectively...In this study, we generated a recombinant adenovirus <b>vaccine</b> encoding the heat-labile enterotoxin B (LTB) and the core neutralizing epitope (COE) of PEDV (rAd-LTB-COE)...The fusion protein LTB-COE was successfully expressed by the recombinant adenovirus in HEK293 cells, and the immunogenicity of the <b>vaccine</b> candidate was assessed in BALB/c mice and piglets...Moreover, a cell-mediated immune response was promoted  in immunized mice, and the neutralizing antibody inhibited both the <b>vaccine</b> strain and the emerging PEDV isolate.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31859605">

Genetic manipulation of porcine deltacoronavirus reveals insights into NS6 and NS7 functions: a novel strategy for vaccine design.

</a>

<small>(PMID31859605</small>)

<br>...In contrast, rPDCoV-DeltaNS6-GFP-infected piglets did not show any clinical signs, indicating that the NS6 protein is an important virulence factor of PDCoV and that the NS6-deficient mutant virus might be a promising live-attenuated <b>vaccine</b> candidate.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31951818">

Influenza A and B in a cohort of outpatient children and adolescent with influenza like-illness during two consecutive influenza seasons.

</a>

<small>(PMID31951818</small>)

<br>...The mismatch between the circulating influenza viruses and the trivalent <b>vaccine</b> offered in Brazil may have contributed to the high frequency of influenza A and B in this population..

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31832902">

Virological and Epidemiological Situation in the Influenza Epidemic Seasons 2016/2017 and 2017/2018 in Poland.

</a>

<small>(PMID31832902</small>)

<br>...As of the 2017/2018 season, a quadrivalent <b>vaccine</b>, consisting of two antigens of influenza A subtypes and another two of influenza B virus, was available in Poland.

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=7><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32264957">

Fighting against the common enemy of COVID-19: a practice of building a community with a shared future for mankind.

</a>

<small>(PMID32264957</small>)

<br>...In order to prevent a potential pandemic-level outbreak of COVID-19, we, as a community of shared future for mankind, recommend for all international leaders to support preparedness in low and middle income countries  especially, take strong global interventions by using old approaches or new tools, mobilize global resources to equip hospital facilities and supplies to protect noisome infections and to provide personal protective tools such as facemask to general population, and quickly initiate research projects on drug and <b>vaccine</b> development.

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32209231">

Computers and viral diseases. Preliminary bioinformatics studies on the design of a synthetic vaccine and a preventative peptidomimetic antagonist against the SARS-CoV-2 (2019-nCoV, COVID-19) coronavirus.

</a>

<small>(PMID32209231</small>)

<br>...This is to find a short section or sections of viral protein sequence suitable for preliminary design proposal for a peptide synthetic <b>vaccine</b> and a peptidomimetic  therapeutic, and to explore some design possibilities...This sequence motif and surrounding variations formed  the basis for proposing a specific synthetic <b>vaccine</b> epitope and peptidomimetic agent.

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32216653">

Managing Oncology Services During a Major Coronavirus Outbreak: Lessons From the  Saudi Arabia Experience.

</a>

<small>(PMID32216653</small>)

<br>...Outbreaks of infectious etiology, particularly those caused by a novel virus that has no known treatment or <b>vaccine</b>, may result in the interruption of medical care provided to patients with cancer and put them at risk for undertreatment in addition to the risk of being exposed to infection, a life-threatening event among patients with cancer.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31712093">

Rapid manipulation of the porcine epidemic diarrhea virus genome by CRISPR/Cas9 technology.

</a>

<small>(PMID31712093</small>)

<br>...Reverse genetics is a valuable tool to study the functions of viral genes and to generate <b>vaccine</b> candidates.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32117569">

Therapeutic strategies in an outbreak scenario to treat the novel coronavirus originating in Wuhan, China.

</a>

<small>(PMID32117569</small>)

<br>...Ultimately, the outbreak could be controlled with a protective <b>vaccine</b> to prevent 2019-nCoV infection...While <b>vaccine</b> research should be pursued intensely, there exists today no therapy to treat 2019-nCoV upon infection, despite an urgent need to find options to help these patients and preclude potential death...Such a treatment could help infected patients before a protective <b>vaccine</b> is developed and widely available in the coming months to year(s)..

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31767069">

Attenuation and characterization of porcine enteric alphacoronavirus strain GDS04 via serial cell passage.

</a>

<small>(PMID31767069</small>)

<br>...Collectively, our research successfully prepared a PEAV attenuated variant which might serve as a live attenuated <b>vaccine</b> candidate against PEAV infection..

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/A/AN/Antibodies, Viral">Antibodies, Viral</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32065055">

Potent binding of 2019 novel coronavirus spike protein by a SARS coronavirus-specific human monoclonal antibody.

</a>

<small>(PMID32065055</small>)

<br>...Currently, however, there is no specific antiviral treatment or <b>vaccine</b>.

<td>Letter</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/V/VA/Vaccines">Vaccines</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31446443">

Advances in Vaccines.

</a>

<small>(PMID31446443</small>)

<br>...Out of 11.2 million children born in EU region, more than 500,000 infants did not receive the complete three-dose series of diphtheria, pertussis, and tetanus <b>vaccine</b> before the first birthday...This chapter provides an overview of recent advances in <b>vaccine</b> development and technologies, manufacturing, characterization of various vaccines, challenges, and strategies in vaccine clinical development.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/R/RE/Respiratory Syncytial Virus Infections">Respiratory Syncytial Virus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32083983">

Molecular characterization of Brazilian wild-type strains of bovine respiratory syncytial virus reveals genetic diversity and a putative new subgroup of the virus.

</a>

<small>(PMID32083983</small>)

<br>...For the BRSV G and F partial gene amplifications, RT-nested-PCR assays were performed with sequencing in both directions with forward and reverse primers used.Results: The G gene-based analysis revealed that two strains were highly similar to the BRSV sequences representative of subgroup III, including the Bayovac <b>vaccine</b> strain.

<td>Journal Article</td>

<td>2020/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/D/DI/Disease Outbreaks">Disease Outbreaks</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31806107">

Zika Vaccine Development: Current Status.

</a>

<small>(PMID31806107</small>)

<br>...In this review, we survey current <b>vaccine</b> efforts, preclinical and clinical results, and ethical and other concerns that directly bear on vaccine development.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/C/CO/Coronavirus">Coronavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32090689">

Trypsin promotes porcine deltacoronavirus mediating cell-to-cell fusion in a cell type-dependent manner.

</a>

<small>(PMID32090689</small>)

<br>...This knowledge can potentially contribute to improvement of virus production efficiency in culture, not only for <b>vaccine</b> preparation but also to develop antiviral treatments..

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/V/VA/Vaccines, Virus-Like Particle">Vaccines, Virus-Like Particle</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31614169">

Preparation of virus-like particle mimetic nanovesicles displaying the S protein  of Middle East respiratory syndrome coronavirus using insect cells.

</a>

<small>(PMID31614169</small>)

<br>...However, to date, no commercial <b>vaccine</b> is available...In this study, structural proteins of MERS-CoV were expressed in silkworm larvae and Bm5 cells for the development of <b>vaccine</b> candidates against MERS-CoV and diagnostic methods.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/P/PL/Plants">Plants</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31542433">

Structure-function and application of plant lectins in disease biology and immunity.

</a>

<small>(PMID31542433</small>)

<br>...Lectins along with heat killed microbes can act as <b>vaccine</b> to provide long term protection from deadly microbes.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/H/HE/Henipavirus Infections">Henipavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31006350">

Nipah virus: epidemiology, pathology, immunobiology and advances in diagnosis, vaccine designing and control strategies - a comprehensive review.

</a>

<small>(PMID31006350</small>)

<br>...High pathogenicity of NiV in humans, and lack of vaccines or therapeutics to counter this disease have attracted attention of researchers worldwide for developing effective NiV <b>vaccine</b> and treatment regimens..

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/E/EN/Enterovirus A, Human">Enterovirus A, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31863580">

A nucleobase-binding pocket in a viral RNA-dependent RNA polymerase contributes to elongation complex stability.

</a>

<small>(PMID31863580</small>)

<br>...Potential applications in antiviral drug and <b>vaccine</b> development are also discussed..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine&from=CORD19#/E/EB/Ebolavirus">Ebolavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31825972">

IL-4/IL-13 polarization of macrophages enhances Ebola virus glycoprotein-dependent infection.

</a>

<small>(PMID31825972</small>)

<br>...No therapeutics or vaccines are currently licensed; however, a <b>vaccine</b> has shown promise in clinical trials.

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 2590 matches before 2019/12

<hr><a name="task4h"><b>Task4h Kaggle Prompt:</b> Efforts to develop animal models and standardize challenge studies</a><p><b>Results:</b><p>

Searching for (challenge study OR challenge OR trial) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=challenge+study+OR+challenge+OR+trial&from=CORD19#/V/VI/Viral Vaccines">Viral Vaccines</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=challenge+study+OR+challenge+OR+trial&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=challenge+study+OR+challenge+OR+trial&from=CORD19#/C/CA/Cattle Diseases">Cattle Diseases</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=challenge+study+OR+challenge+OR+trial&from=CORD19#/C/CH/Chemokine CCL2">Chemokine CCL2</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=challenge+study+OR+challenge+OR+trial&from=CORD19#/C/CR/Croup">Croup</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=challenge+study+OR+challenge+OR+trial&from=CORD19#/V/VI/Viral Vaccines">Viral Vaccines</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31950289">

Recombinant adenovirus carrying a core neutralizing epitope of porcine epidemic diarrhea virus and heat-labile enterotoxin B of Escherichia coli as a mucosal vaccine.

</a>

<small>(PMID31950289</small>)

<br>...Further studies are required to evaluate the efficacy of rAd-LTB-COE against a highly virulent PEDV <b><b>challenge</b></b>..

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=8><a href="http://www.softconcourse.com/CORD19/?filterText=challenge+study+OR+challenge+OR+trial&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32231345">

Inhibition of SARS-CoV-2 (previously 2019-nCoV) infection by a highly potent pan-coronavirus fusion inhibitor targeting its spike protein that harbors a high  capacity to mediate membrane fusion.

</a>

<small>(PMID32231345</small>)

<br>...Intranasal application of EK1C4 before or after <b><b>challenge</b></b> with HCoV-OC43 protected mice from infection, suggesting that EK1C4 could be used for prevention and treatment of infection by the currently circulating SARS-CoV-2 and other emerging SARSr-CoVs..

<td>Journal Article</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32264908">

First statement on preparation for the COVID-19 pandemic in large German Speaking University-based radiation oncology departments.

</a>

<small>(PMID32264908</small>)

<br>...Communication with our neighboring countries, within societies and between departments can help meet the <b><b>challenge</b></b>... Here, we report on our learning system and preparation measures to effectively tackle the COVID-19 <b><b>challenge</b></b> in University-Based Radiation Oncology Departments..

<td>Journal Article; Review</td>

<td>2020/04</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32155789">

Immediate Psychological Responses and Associated Factors during the Initial Stage of the 2019 Coronavirus Disease (COVID-19) Epidemic among the General Population  in China.

</a>

<small>(PMID32155789</small>)

<br>...Background: The 2019 coronavirus disease (COVID-19) epidemic is a public health emergency of international concern and poses a <b><b>challenge</b></b> to psychological resilience.

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32209164">

Coronavirus disease (COVID-19) in a paucisymptomatic patient: epidemiological and clinical challenge in settings with limited community transmission, Italy, February 2020.

</a>

<small>(PMID32209164</small>)

<br>....

<td>Journal Article</td>

<td>2020/03</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32040190">

A qualitative study of zoonotic risk factors among rural communities in southern  China.

</a>

<small>(PMID32040190</small>)

<br>...However, the risk factors leading  to emergence are poorly understood, which presents a <b><b>challenge</b></b> in developing appropriate mitigation strategies for local communities.

<td>Journal Article; Research Support, N.I.H., Extramural; Research Support, Non-U.S. Gov't; Research Support, U.S. Gov't, Non-P.H.S.</td>

<td>2020/02</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31992387">

Detection of 2019 novel coronavirus (2019-nCoV) by real-time RT-PCR.

</a>

<small>(PMID31992387</small>)

<br>...BACKGROUND: The ongoing outbreak of the recently emerged novel coronavirus (2019-nCoV) poses a <b><b>challenge</b></b> for public health laboratories as virus isolates are unavailable while there is growing evidence that the outbreak is more widespread than initially thought, and international spread through travellers does already occur.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32226291">

Progression of Mental Health Services during the COVID-19 Outbreak in China.

</a>

<small>(PMID32226291</small>)

<br>...Nevertheless, the rapid transmission of the COVID-19 has emerged to mount a serious <b><b>challenge</b></b> to the mental health service in China..

<td>Journal Article; Research Support, Non-U.S. Gov't; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=challenge+study+OR+challenge+OR+trial&from=CORD19#/C/CA/Cattle Diseases">Cattle Diseases</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31548072">

Effect of antibiotic treatment in preweaned Holstein calves after experimental bacterial challenge with Pasteurella multocida.

</a>

<small>(PMID31548072</small>)

<br>...The primary objective of this randomized controlled <b>challenge</b> <b>study</b> was to investigate the effect of ampicillin on ultrasonographic (US) lung consolidation  after experimental challenge with Pasteurella multocida in preweaned dairy calves...once daily for 3 d] when >/=1 cm(2) of lung consolidation was observed and >/=6 h had elapsed since <b><b>challenge</b></b>.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=challenge+study+OR+challenge+OR+trial&from=CORD19#/C/CH/Chemokine CCL2">Chemokine CCL2</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31710162">

Tumor-associated macrophages secrete CC-chemokine ligand 2 and induce tamoxifen resistance by activating PI3K/Akt/mTOR in breast cancer.

</a>

<small>(PMID31710162</small>)

<br>...Although endocrine therapy is effective, the development of endocrine resistance is a major clinical <b><b>challenge</b></b>.

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=challenge+study+OR+challenge+OR+trial&from=CORD19#/C/CR/Croup">Croup</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31542653">

Defining atypical croup: A case report and review of the literature.

</a>

<small>(PMID31542653</small>)

<br>...The clinical course of croup is well-described, however atypical presentations pose a diagnostic and management <b><b>challenge</b></b>.

<td>Case Reports; Journal Article; Systematic Review</td>

<td>2019/12</td>

</tr>

</table>

<p>There are also 1435 matches before 2019/12

<hr><a name="task4i"><b>Task4i Kaggle Prompt:</b> Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers</a><p><b>Results:</b><p>

Searching for (prophylaxis OR prophylactic) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=prophylaxis+OR+prophylactic&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=prophylaxis+OR+prophylactic&from=CORD19#/A/AN/Antibodies, Monoclonal">Antibodies, Monoclonal</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=prophylaxis+OR+prophylactic&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=prophylaxis+OR+prophylactic&from=CORD19#/A/AN/Antiviral Agents">Antiviral Agents</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32054787">

Prophylactic and therapeutic remdesivir (GS-5734) treatment in the rhesus macaque model of MERS-CoV infection.

</a>

<small>(PMID32054787</small>)

<br>...Here, we tested the efficacy of <b>prophylactic</b> and therapeutic remdesivir treatment in a nonhuman primate model of  MERS-CoV infection, the rhesus macaque...<b>Prophylactic</b> remdesivir treatment initiated 24 h prior to inoculation completely prevented MERS-CoV-induced clinical disease, strongly inhibited MERS-CoV replication in respiratory tissues, and prevented the formation of lung lesions.

<td>Journal Article; Research Support, N.I.H., Intramural</td>

<td>2020/03</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=prophylaxis+OR+prophylactic&from=CORD19#/A/AN/Antibodies, Monoclonal">Antibodies, Monoclonal</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31526954">

Detection of MERS-CoV antigen on formalin-fixed paraffin-embedded nasal tissue of alpacas by immunohistochemistry using human monoclonal antibodies directed against different epitopes of the spike protein.

</a>

<small>(PMID31526954</small>)

<br>...In recent years, several investigators developed protective antibodies which could be used  as <b>prophylaxis</b> in prospective human epidemics.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=3><a href="http://www.softconcourse.com/CORD19/?filterText=prophylaxis+OR+prophylactic&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31924756">

Comparative therapeutic efficacy of remdesivir and combination lopinavir, ritonavir, and interferon beta against MERS-CoV.

</a>

<small>(PMID31924756</small>)

<br>...In mice, both <b>prophylactic</b> and therapeutic RDV improve pulmonary function and reduce lung viral loads and severe lung pathology...In contrast, <b>prophylactic</b> LPV/RTV-IFNb slightly reduces viral loads without impacting other disease parameters.

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2020/01</td>

</tr>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32005086">

An emerging coronavirus causing pneumonia outbreak in Wuhan, China: calling for developing therapeutic and prophylactic strategies.

</a>

<small>(PMID32005086</small>)

<br>....

<td>Journal Article</td>

<td>2020</td>

</tr>

</table>

<p>There are also 520 matches before 2019/12

<hr><a name="task4j"><b>Task4j Kaggle Prompt:</b> Approaches to evaluate risk for enhanced disease after vaccination</a><p><b>Results:</b><p>

Searching for (vaccine risk) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+risk&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+risk&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+risk&from=CORD19#/I/IN/Influenza, Human">Influenza, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31951818">

Influenza A and B in a cohort of outpatient children and adolescent with influenza like-illness during two consecutive influenza seasons.

</a>

<small>(PMID31951818</small>)

<br>....

<td>Journal Article</td>

<td>2020/01</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+risk&from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32216653">

Managing Oncology Services During a Major Coronavirus Outbreak: Lessons From the  Saudi Arabia Experience.

</a>

<small>(PMID32216653</small>)

<br>...Outbreaks of infectious etiology, particularly those caused by a novel virus that has no known treatment or <b>vaccine</b>, may result in the interruption of medical care provided to patients with cancer and put them at <b>risk</b> for undertreatment in addition to the risk of being exposed to infection, a life-threatening event among patients with cancer.

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/03</td>

</tr>

</table>

<p>There are also 209 matches before 2019/12

<hr><a name="task4k"><b>Task4k Kaggle Prompt:</b> Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]</a><p><b>Results:</b><p>

Searching for (vaccine development) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings

<blockquote>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/V/VI/Viral Vaccines">Viral Vaccines</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/V/VA/Vaccines">Vaccines</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/A/AN/Antibodies, Viral">Antibodies, Viral</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/V/VA/Vaccines, Virus-Like Particle">Vaccines, Virus-Like Particle</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/Z/ZI/Zika Virus Infection">Zika Virus Infection</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/E/EB/Ebolavirus">Ebolavirus</a></span>

<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/E/EN/Enterovirus A, Human">Enterovirus A, Human</a></span>

</blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.

<table class=l><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/V/VI/Viral Vaccines">Viral Vaccines</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31859605">

Genetic manipulation of porcine deltacoronavirus reveals insights into NS6 and NS7 functions: a novel strategy for vaccine design.

</a>

<small>(PMID31859605</small>)

<br>....

<td>Journal Article</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/V/VA/Vaccines">Vaccines</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31446443">

Advances in Vaccines.

</a>

<small>(PMID31446443</small>)

<br>...This chapter provides an overview of recent advances in <b>vaccine</b> <b>development</b> and technologies, manufacturing, characterization of various vaccines, challenges, and strategies in vaccine clinical development.

<td>Journal Article; Review</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32264957">

Fighting against the common enemy of COVID-19: a practice of building a community with a shared future for mankind.

</a>

<small>(PMID32264957</small>)

<br>...In order to prevent a potential pandemic-level outbreak of COVID-19, we, as a community of shared future for mankind, recommend for all international leaders to support preparedness in low and middle income countries  especially, take strong global interventions by using old approaches or new tools, mobilize global resources to equip hospital facilities and supplies to protect noisome infections and to provide personal protective tools such as facemask to general population, and quickly initiate research projects on drug and <b>vaccine</b> <b>development</b>.

<td>Letter</td>

<td>2020/04</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/A/AN/Antibodies, Viral">Antibodies, Viral</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/32065055">

Potent binding of 2019 novel coronavirus spike protein by a SARS coronavirus-specific human monoclonal antibody.

</a>

<small>(PMID32065055</small>)

<br>....

<td>Letter</td>

<td>2020</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/V/VA/Vaccines, Virus-Like Particle">Vaccines, Virus-Like Particle</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31614169">

Preparation of virus-like particle mimetic nanovesicles displaying the S protein  of Middle East respiratory syndrome coronavirus using insect cells.

</a>

<small>(PMID31614169</small>)

<br>...In this study, structural proteins of MERS-CoV were expressed in silkworm larvae and Bm5 cells for the <b>development</b> of <b>vaccine</b> candidates against MERS-CoV and diagnostic methods.

<td>Journal Article</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/Z/ZI/Zika Virus Infection">Zika Virus Infection</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31806107">

Zika Vaccine Development: Current Status.

</a>

<small>(PMID31806107</small>)

<br>...In this review, we survey current <b>vaccine</b> efforts, preclinical and clinical results, and ethical and other concerns that directly bear on vaccine <b>development</b>.

<td>Journal Article; Review</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/E/EB/Ebolavirus">Ebolavirus</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31825972">

IL-4/IL-13 polarization of macrophages enhances Ebola virus glycoprotein-dependent infection.

</a>

<small>(PMID31825972</small>)

<br>....

<td>Journal Article; Research Support, N.I.H., Extramural</td>

<td>2019/12</td>

</tr>

<tr valign=top><td rowspan=2><a href="http://www.softconcourse.com/CORD19/?filterText=vaccine+development&from=CORD19#/E/EN/Enterovirus A, Human">Enterovirus A, Human</a>

<tr valign=top><td>

 <a  target=_blank href="https://pubmed.ncbi.nlm.nih.gov/31863580">

A nucleobase-binding pocket in a viral RNA-dependent RNA polymerase contributes to elongation complex stability.

</a>

<small>(PMID31863580</small>)

<br>...Potential applications in antiviral drug and <b>vaccine</b> <b>development</b> are also discussed..

<td>Journal Article; Research Support, Non-U.S. Gov't</td>

<td>2020/02</td>

</tr>

</table>

<p>There are also 980 matches before 2019/12



"""



h = display(HTML(htmlresults))

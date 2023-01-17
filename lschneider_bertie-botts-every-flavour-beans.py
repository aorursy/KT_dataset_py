#install pytextrank if required
%pip install pytextrank

import spacy
import pytextrank
from math import sqrt
from operator import itemgetter

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import csv
import pandas

!pip install sentence-transformers

from sentence_transformers import SentenceTransformer

import scipy
import itertools

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
from pylab import rcParams

!pip install bert-extractive-summarizer
!pip install transformers==2.8.0
!pip install neuralcoref
!python -m spacy download en_core_web_md

from IPython.display import clear_output

clear_output()
import os
import json

paperId2Abstract = {}
paperId2PaperText = {}

pathToFile = "../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/"

listOfFiles = os.listdir(pathToFile)
numberOfFiles = len(listOfFiles)
counter = 1

for filename in os.listdir(pathToFile):
    if filename.endswith(".json") : 
      print(round(counter/numberOfFiles*100,1), "%")
      with open(pathToFile+filename) as f:
        try:
          data = json.load(f)
          paperId = data["paper_id"]
          title = data["metadata"]["title"]
          abstract = data["abstract"]
          body = data["body_text"]

          paperText = title + "\n"

          counter = counter + 1
          
          if (len(abstract)>0):
            #only consider first chunk here
            paperId2Abstract[paperId] = abstract[0]["text"]

            for abstractChunk in abstract:
              paperText += abstractChunk["text"]+"\n"
          else:
            print("no abstract available")

          if  (len(body)>0):
            for bodyChunk in body:
              paperText += bodyChunk["text"]+"\n"
          else:
            print("no body text available")

          paperId2PaperText[paperId] = paperText
          
        except:
            print("an exception occurred")
#create a list of summary sentences as input for BERT
sentences = list(paperId2Abstract.values())
#write dictionaries to file
f = open("paperId2Abstract.csv", "w")
w = csv.writer(f)
for key, val in paperId2Abstract.items():
    w.writerow([key, val])
f.close()

f2 = open("paperId2PaperText.csv", "w")
w2 = csv.writer(f2)
for key, val in paperId2PaperText.items():
    w2.writerow([key, val])
f2.close()

#write summary sentences to file
with open('sentences.txt', 'w') as f:
    for item in sentences:
        f.write("%s\n" % item)
def summarize_stops_removed(inputText, numberOfSummarySentences):

  output=[]

  #load a spaCy model, depending on language, scale, etc.
  nlp = spacy.load("en_core_web_sm")

  #add PyTextRank to the spaCy pipeline
  tr = pytextrank.TextRank()
  nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

  doc = nlp(inputText)
  
  sent_bounds = [ [s.start, s.end, set([])] for s in doc.sents ]

  limit_phrases = numberOfSummarySentences

  phrase_id = 0
  unit_vector = []

  for p in doc._.phrases:
    unit_vector.append(p.rank)
    for chunk in p.chunks:  
      for sent_start, sent_end, sent_vector in sent_bounds:
        if chunk.start >= sent_start and chunk.start <= sent_end:
          sent_vector.add(phrase_id)
          break

    phrase_id += 1
    
    if phrase_id == limit_phrases:
      break
  
  sum_ranks = sum(unit_vector)
  unit_vector = [ rank/sum_ranks for rank in unit_vector ]

  sent_rank = {}
  sent_id = 0

  for sent_start, sent_end, sent_vector in sent_bounds:
    #print(sent_vector)
    sum_sq = 0.0
    for phrase_id in range(len(unit_vector)):
      if phrase_id not in sent_vector:
        sum_sq += unit_vector[phrase_id]**2.0

    sent_rank[sent_id] = sqrt(sum_sq)
    sent_id += 1

  sorted(sent_rank.items(), key=itemgetter(1))

  limit_sentences = numberOfSummarySentences

  sent_text = {}
  sent_id = 0

  for sent in doc.sents:
    sent_text[sent_id] = sent.text
    sent_id += 1

  num_sent = 0

  for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):
    output.append(sent_text[sent_id])

    num_sent += 1
    
    if num_sent > limit_sentences:
      break
  
    final_output=''.join(map(str, output))
    
    list = stopwords.words('english')
    StopsRemoved = [word for word in final_output.split() if word not in list]
    StopsRemoved_str = ' '.join(map(str, StopsRemoved))

  return StopsRemoved_str
MERS='Middle East Respiratory Syndrome (MERS) is a zoonotic viral disease that can be transmitted from dromedaries to human beings. More than 1500 cases of MERS have been reported in human beings to date. Although MERS has been associated with 30% case fatality in human beings, MERS coronavirus (MERS-CoV) infection in dromedaries is usually asymptomatic. In rare cases, dromedaries may develop mild respiratory signs. No MERS-CoV or antibodies against the virus have been detected in camelids other than dromedaries. MERS-CoV is mainly acquired in dromedaries when they are less than 1 year of age, and the proportion of seropositivity increases with age to a seroprevalence of 100% in adult dromedaries. Dromedary or one humped camels (Camelus dromedarius) are so far the only reservoir of MERS-CoV. Adult dromedaries have almost 100% seropositivity against MERS-CoV while the virus is found mainly in dromedary calves. Camel-to-human transmission of MERS-CoV and subsequent human-to-human transmission result in MERS in human beings, many of whom develop severe lower respiratory tract infections, with renal failure in some cases. Laboratory diagnosis of MERS-CoV infection in dromedaries can be achieved through virus isolation using Vero cells, RNA detection by real-time quantitative reverse transcriptase-PCR and antigen detection using respiratory specimens or serum. Rapid nucleocapsid antigen detection using a lateral flow platform allows efficient screening of dromedaries carrying MERS-CoV. In addition to MERS-CoV, which is a lineage C virus in the Betacoronavirus (betaCoV) genus, a lineage B betaCoV and a virus in the Alphacoronavirus (alphaCoV) genus have been detected in dromedaries. Dromedary CoV UAE-HKU23 is closely related to human CoV OC43, whereas the alphaCoV has not been detected in human beings to date. Human coronaviruses cause both upper and lower respiratory tract infections in humans. MERS-CoV, a lineage C Betacoronavirus (βCoVs), has a positive-sense single-stranded RNA (ssRNA) genome about 30-kb in size. The MERS-CoV genomes share more than 99% sequence identity, indicating a low mutation rate and low variance among the genomes. MERS-CoV has also evolved mechanisms to evade the host immune system. In 2012, a sixth human coronavirus (hCoV) was isolated from a patient presenting with severe respiratory illness. The 60-year-old man died as a result of renal and respiratory failure after admission to a hospital in Jeddah, Saudi Arabia. The aetiological agent was eventually identified as a coronavirus and designated Middle East respiratory syndrome coronavirus (MERS-CoV). Chest radiography and computed tomography findings are generally consistent with viral pneumonitis and acute respiratory distress syndrome. Laboratory findings include lymphopenia, thrombocytopenia and elevated lactate dehydrogenase levels, with some cases with a consumptive coagulopathy and elevations in creatinine, lactate dehydrogenase and liver enzymes.The clinical spectrum of MERS-CoV infection ranges from asymptomatic infection to rapidly progressive, acute respiratory distress syndrome, septic shock and multi-organ failure and death. Initial symptoms are often nonspecific and patients report general malaise, including low grade fever, chills, headache, nonproductive cough, dyspnea, and myalgia. Other symptoms can include sore throat and MERS-CoV patients can also present with gastrointestinal symptoms such as anorexia, nausea and vomiting, abdominal pain and diarrhea. Atypical presentations, including mild respiratory illness without fever and diarrheal illness, preceding the development of pneumonia have been documented. Up to 50% of adult symptomatic patients require intensive care unit (ICU) treatment. These patients often show no sign of improvement and 40–70% typically require mechanical ventilation within the first week. Renal replacement therapy is required for between 22–70% of critically ill patients. Neuromuscular complications are not rare during MERS treatment, and could simply have been underdiagnosed previously. Severe neurological syndrome, characterized by varying degrees of disturbed consciousness, ataxia, focal motor deficit and bilateral hyper-intense lesions were reported from a retrospective study of patients in ICU. MERS-CoV can be detected in respiratory tract secretions, with tracheal secretions and broncho-alveolar lavage specimens containing a higher viral load than nasopharyngeal swabs. The virus has also been detected in feces, serum and urine. Virus excretion peaks approximately 10 days after the onset of symptoms, but viable viruses can be shed through respiratory secretions for up to 25 days from clinically fully recovered patients. MERS-CoV strains from human beings are polyphyletic as a result of multiple camel-to-human transmission events. The low frequency of transmission from camels to human beings is due to the fact that only young dromedaries with no or low maternal antibodies to MERS-CoV are susceptible to infection and the virus is shed only for 8 days. These young dromedaries are reared with their mothers for a year and have no or very little contact with human beings. Additionally, less than 1% of infected calves exhibit nasal discharge and therefore the quantity of virus excreted may be low. Another reason for the low transmission rate is likely to be the lack expression of the MERS-CoV receptor-dipeptidyl peptidase 4 (DPP4) in the human upper respiratory tract. MERS-CoV has now been reported in more than 27 countries across the Middle East, Europe, North Africa and Asia. As of July 2017, 2040 MERS-CoV laboratory confirmed cases, resulting in 712 deaths, were reported globally, with a majority of these cases from the Arabian Peninsula. This review summarises the current understanding of MERS-CoV, with special reference to the (i) genome structure; (ii) clinical features; (iii) diagnosis of infection; and (iv) treatment and vaccine development. CoVs have been linked to several outbreaks of diarrhoea affecting New World Camelids of all ages on farms in north west USA and South America. More ‘new’ CoVs are likely to be generated due to the high recombination rate in this group of viruses. More intensive surveillance of CoVs in camels should be performed to improve understanding of these viruses in this unique group of animals. We successfully isolated DcCoV UAE-HKU23 using the human rectal tumour HRT-18G cell line and confirmed that it is different from the betaCoV in alpacas using bioinformatics approaches. Although both MERS-CoV and DcCoV UAE-HKU23 are beta CoVs, there is minimal cross antigenicity between them, as shown by various serological tests.. Another alphaCoV that has been found in alpacas was also discovered in nasal samples of dromedaries. Protective experimental immunisations in dromedaries have already started using a modified vaccinia virus Ankara (MVA) vaccine expressing the MERS-CoV spike protein. A significant reduction in excretion of infectious virus and viral RNA in small numbers of vaccinated and challenged dromedaries was observed compared to controls. Protection is correlated with the presence of serum neutralising antibodies against MERS-CoV. The experimentally infected animals demonstrated moderate rhinitis, with nasal discharge, tracheitis and bronchitis, but no involvement of the alveolar tissue. The dromedaries were infected with high doses of MERS-CoV intranasally, i.e. 1 × 10e7 50% tissue culture infectious doses (TCID50) and 5 × 10e6 TCID50. Experimental MERS-CoV infections were also performed in alpacas, with similar results. Experimentally infected alpacas transmitted the virus to two of three contact animals. Experimentally infected animals were protected against reinfection 70 days later and those infected by contact were only partially protected.'
SARS='Severe acute respiratory syndrome (SARS) occurring in China were caused by coronaviruses (CoVs): SARS-CoV. The virus utilize the angiotensin-converting enzyme 2 (ACE2) receptor for invading human bodies. The epidemics occurred in cold dry winter seasons celebrated with major holidays, and started in regions where dietary consumption of wildlife is a fashion. Thus, if bats were the natural hosts of SARS-CoV, cold temperature and low humidity in these times might provide conducive environmental conditions for prolonged viral survival in these regions concentrated with bats. The widespread existence of this bat-carried or -released virus might have an easier time in breaking through human defenses when harsh winter makes human bodies more vulnerable. Once succeeding in making some initial human infections, spreading of the disease was made convenient with increased social gathering and holiday travel. These natural and social factors influenced the general progression and trajectory of the SARS epidemiology. At the end of 2002, the first cases of severe acute respiratory syndrome (SARS) were reported, and in the following year, SARS resulted in considerable mortality and morbidity worldwide. SARS is caused by a novel species of coronavirus (SARS-CoV) and is the most severe coronavirus-mediated human disease that has been described so far. On the basis of similarities with other coronavirus infections, SARS might, in part, be immune mediated. Studies of animals that are infected with other coronaviruses indicate that excessive and sometimes dysregulated responses by macrophages and other pro-inflammatory cells might be particularly important in the pathogenesis of disease that is caused by infection with these viruses. Lessons from such studies will help us to understand more about the pathogenesis of SARS in humans and to prevent or control outbreaks of SARS in the future. SARS-CoV, which causes a severe respiratory disease, seems to be an enzootic virus in Southeast Asia. Several species that might be infected, such as masked palm civets (Paguma larvata), are consumed as food in parts of China, and the wet markets, at which live animals are bought and sold, are likely venues for the initial crossover event to humans. The 2002–2003 outbreak of SARS in humans probably resulted from an interspecific transfer of the virus by aerosols from live, exotic animals that were infected with SARS-CoV to workers in these wet markets. Sera from masked palm civets, raccoon dogs (Nyctereutes procyonoides) and Chinese ferret-badgers (Melogale moschata) were shown to contain neutralizing antibodies that were specific for SARS-CoV, and virus that was nearly identical to the strains that were isolated from infected humans was detected in masked palm civets. Infection of humans with SARS-CoV typically causes an influenza-like syndrome of malaise, rigors, fatigue and high fevers. In two-thirds of infected patients, the disease progresses to an atypical pneumonia, with shortness of breath and poor oxygen exchange in the alveoli. Many of these patients also develop watery diarrhoea with active virus shedding, which might increase the transmissibility of the virus. Respiratory insufficiency leading to respiratory failure is the most common cause of death among those infected with SARS-CoV. Consistent with these clinical observations, the host cell-surface receptor for SARS-CoV, angiotensin-converting enzyme 2 (ACE2), is detected in the lungs and gastrointestinal tract. Severe cases of SARS are associated with lymphopaenia, neutrophilia, mild thrombocytopaenia and coagulation defects. Haemophagocytosis, which is indicative of cytokine dysregulation, is also detected in some patients with severe disease. Damage to the lungs of patients who are infected with SARS-CoV seems to occur directly, by viral destruction of alveolar and bronchial epithelial cells and macrophages, as well as indirectly, through production of immune mediators, although the exact role of these direct and indirect mechanisms remains controversial. Viral load, as determined from titres in nasopharyngeal aspirate, diminishes 10–15 days after the onset of symptoms, even though clinical disease and alveolar damage worsen, indicating that the host immune response is responsible for some of the pathology in SARS-CoV-infected patients. High concentrations of virus have been detected in several organs at autopsy, including the lungs, intestine, kidneys and brain. SARS-CoV directly infects T cells, contributing to lymphopaenia and to atrophy of the spleen and lymphoid tissue. SARS-CoV also interferes with the initiation of the innate immune response by inhibiting the expression of type I interferons (IFNs) by infected cells, including human monocyte-derived dendritic cells (DCs) and macrophages. IFN production requires the phosphorylation and dimerization of a constitutively expressed protein, IFN-regulatory factor 3 (IRF3). IRF3 is not activated, at least in vitro, after infection with SARS-CoV. Consistent with a role for CXCL8 in pathogenesis, severe disease is associated with an increase in the number of neutrophils in the blood. Increased serum concentrations of two anti-inflammatory molecules — transforming growth factor-β and prostaglandin E2 — were detected. In patients with SARS, virus is detected in the lungs and in immune cells at the time of death, indicating that virus directly causes pulmonary and immune-system injury.SARS-CoV-infected macrophages and DCs express increased amounts of pro-inflammatory cytokines. Consistent with this, increased concentrations of pro-inflammatory chemokines and other cytokines are present in most infected patients. The 9% mortality was somewhat higher than the expected mortality for the usual causes of community-acquired pneumonias (approximately 2–4%), the 50% mortality for patients over age 60 also distinguished this pathogen. Several analyses pointed to older age as an independent risk factor for dying. SARS to many clinicians appeared to have a biphasic course, with cough and fever initially followed in 3–5 days with a normal temperature and increasing hypoxia. Before the notification from Vietnam of SARS, the global web-based surveillance system overseen by the International Society for Infectious Diseases was suggesting a new epidemic. The World Health Organization responded admirably to the SARS epidemic, taking international leadership in addressing questions to the public, coordinating scientific investigations, and quickly reporting all new advances from the laboratory and field epidemiological studies. The culinary delicacies of southern China—what we call exotic food choices—led initially to infections in animal handlers, chefs, and caterers and subsequently had a huge impact on the lives of people thousands of miles away. They were aware of the risk of infection, the possibility of serious morbidity and even of dying from infection transmitted in the hospital as an occupational hazard. With SARS, the word quarantine applies to those people—including health care workers—who were identified to have been exposed but had not yet shown any signs of illness. They would be expected to remain secluded, usually at home, until the incubation period—usually 2–8 days—is exceeded. The SARS coronavirus, which has a crown-like appearance when seen through the electron microscope lept from animals to nearby people—exotic animal handlers, chefs, and caterers initially. Close contact is required for human transmission: the victim has to be within 3 ft of a patient to transfer the virus via a large droplet. The virus is also found in saliva, urine, sweat and tears of infected people, is shed in the feces for 30 days and can survive for over 24 h on hard surfaces.The versatile SARS coronavirus has stunned the economic and health care institutions around the world.We would summarize by emphasizing the following: the importance of clinicians recognizing a new syndrome, the need for individuals and countries to report epidemics, the role of information technology to communicate, and the key role for the WHO. Quarantine must be employed with care and compassion.'
COVID19='Coronavirus disease 2019 (COVID-19) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2).The disease was first identified in December 2019 in Wuhan, the capital of Chinas Hubei province, and has since spread globally, resulting in the ongoing 2019–20 coronavirus pandemic.Common symptoms include fever, cough and shortness of breath.Other symptoms may include fatigue, muscle pain, diarrhea, sore throat, loss of smell and abdominal pain.The time from exposure to onset of symptoms is typically around five days, but may range from two to 14 days.While the majority of cases result in mild symptoms, some progress to viral pneumonia and multi-organ failure. As of 15 April 2020, more than 1.98 million cases have been reported across 210 countries and territories,resulting in over 126,000 deaths. More than 486,000 people have recovered.The virus is mainly spread during close contact and by small droplets produced when those infected cough, sneeze, or talk. These droplets may also be produced during breathing; however, they rapidly fall to the ground or surfaces and are not generally spread through the air over large distances. People may also become infected by touching a contaminated surface and then their face.The virus can survive on surfaces for up to 72 hours.Coronavirus is most contagious during the first three days after onset of symptoms, although spread may be possible before symptoms appear and in later stages of the disease.The standard method of diagnosis is by real-time reverse transcription polymerase chain reaction (rRT-PCR) from a nasopharyngeal swab. The infection can also be diagnosed from a combination of symptoms, risk factors and a chest CT scan showing features of pneumonia. Recommended measures to prevent infection include frequent hand washing, social distancing (maintaining physical distance from others, especially from those with symptoms), covering coughs and sneezes with a tissue or inner elbow and keeping unwashed hands away from the face.The use of masks is recommended for those who suspect they have the virus and their caregivers. Recommendations for mask use by the general public vary, with some authorities recommending against their use, some recommending their use and others requiring their use. Management involves treatment of symptoms, supportive care, isolation and experimental measures.The World Health Organization (WHO) declared the 2019–20 coronavirus outbreak a Public Health Emergency of International Concern (PHEIC) on 30 January 2020 and a pandemic on 11 March 2020.Local transmission of the disease has been recorded in many countries across all six WHO regions.Those infected with the virus may be asymptomatic or develop flu-like symptoms such as fever, cough, fatigue, and shortness of breath. Emergency symptoms include difficulty breathing, persistent chest pain or pressure, confusion, difficulty waking, and bluish face or lips; immediate medical attention is advised if these symptoms are present. Less commonly, upper respiratory symptoms—such as sneezing, runny nose, or sore throat—may be seen. Gastrointestinal symptoms such as nausea, vomiting and diarrhoea have been observed in varying percentages. Some cases in China initially presented only with chest tightness and palpitations. In March 2020, reports emerged indicating that loss of the sense of smell (anosmia) may be a common symptom among those with mild cases. In some, the disease may progress to pneumonia, multi-organ failure, and death. In those who develop severe symptoms, time from symptom onset to needing mechanical ventilation is typically eight days. As is common with infections, there is a delay between the moment when a person is infected with the virus and the time when they develop symptoms. This is called the incubation period. The incubation period for COVID-19 is typically five to six days but may range from two to 14 days. 97.5% of people who develop symptoms will do so within 11.5 days of infection. Reports indicate that not all who are infected develop symptoms. The role of these asymptomatic carriers in transmission contribute to the spread of the disease. The proportion of infected people who do not display symptoms is currently unknown and being studied, with the Korea Centers for Disease Control and Prevention (KCDC) reporting that 20% of all confirmed cases remained asymptomatic during their hospital stay. Chinas National Health Commission began including asymptomatic cases in its daily cases on 1 April; of the 166 infections on that day, 130 (78%) were asymptomatic. It was observed that the virus was present in most patients saliva in quantities reaching 100 million virus strands per 1 mL. SARS-CoV-2 is closely related to the original SARS-CoV. It is thought to have a zoonotic origin. Genetic analysis has revealed that the coronavirus genetically clusters with the genus Betacoronavirus, in subgenus Sarbecovirus (lineage B) together with two bat-derived strains. It is 96% identical at the whole genome level to other bat coronavirus samples (BatCov RaTG13). In February 2020, Chinese researchers found that there is only one amino acid difference in certain parts of the genome sequences between the viruses from pangolins and those from humans; however, whole-genome comparison to date found that at most 92% of genetic material was shared between pangolin coronavirus and SARS-CoV-2, which is insufficient to prove pangolins to be the intermediate host. The lungs are the organs most affected by COVID-19 because the virus accesses host cells via the enzyme angiotensin-converting enzyme 2 (ACE2), which is most abundant in the type II alveolar cells of the lungs. The virus uses a special surface glycoprotein called a "spike" (peplomer) to connect to ACE2 and enter the host cell. The density of ACE2 in each tissue correlates with the severity of the disease in that tissue and some have suggested that decreasing ACE2 activity might be protective, though another view is that increasing ACE2 using angiotensin II receptor blocker medications could be protective and these hypotheses need to be tested. As the alveolar disease progresses, respiratory failure might develop and death may follow. The virus also affects gastrointestinal organs as ACE2 is abundantly expressed in the glandular cells of gastric, duodenal and rectal epithelium as well as endothelial cells and enterocytes of the small intestine. Autopsies of people who died of COVID-19 have found diffuse alveolar damage (DAD), and lymphocyte-containing inflammatory infiltrates within the lung. Although SARS-COV-2 has a tropism for ACE2-expressing epithelial cells of the respiratory tract, patients with severe COVID-19 have symptoms of systemic hyperinflammation. Clinical laboratory findings of elevated IL-2, IL-7, IL-6, granulocyte-macrophage colony-stimulating factor (GM-CSF), interferon-γ inducible protein 10 (IP-10), monocyte chemoattractant protein 1 (MCP-1), macrophage inflammatory protein 1-α (MIP-1α), and tumour necrosis factor-α (TNF-α) indicative of cytokine release syndrome (CRS) suggest an underlying immunopathology. Additionally, people with COVID-19 and acute respiratory distress syndrome (ARDS) have classical serum biomarkers of CRS including elevated C-reactive protein (CRP), lactate dehydrogenase (LDH), D-dimer, and ferritin. Systemic inflammation results in vasodilation, allowing inflammatory lymphocytic and monocytic infiltration of the lung and the heart. In particular, pathogenic GM-CSF-secreting T-cells were shown to correlate with the recruitment of inflammatory IL-6-secreting monocytes and severe lung pathology in COVID-19 patients. Lymphocytic infiltrates have also been reported at autopsy. People are managed with supportive care, which may include fluid therapy, oxygen support, and supporting other affected vital organs. The CDC recommends that those who suspect they carry the virus wear a simple face mask. Extracorporeal membrane oxygenation (ECMO) has been used to address the issue of respiratory failure, but its benefits are still under consideration. The WHO and Chinese National Health Commission have published recommendations for taking care of people who are hospitalised with COVID-19. Intensivists and pulmonologists in the U.S. have compiled treatment recommendations from various agencies into a free resource, the IBCC. As of April 2020, there is no specific treatment for COVID-19. For symptoms, some medical professionals recommend paracetamol (acetaminophen) over ibuprofen for first-line use. The WHO does not oppose the use of non-steroidal anti-inflammatory drugs (NSAIDs) such as ibuprofen for symptoms, and the FDA says currently there is no evidence that NSAIDs worsen COVID-19 symptoms. While theoretical concerns have been raised about ACE inhibitors and angiotensin receptor blockers, as of 19 March 2020, these are not sufficient to justify stopping these medications. Steroids, such as methylprednisolone, are not recommended unless the disease is complicated by acute respiratory distress syndrome. Precautions must be taken to minimise the risk of virus transmission, especially in healthcare settings when performing procedures that can generate aerosols, such as intubation or hand ventilation. For healthcare professionals caring for people with COVID-19, the CDC recommends placing the person in an Airborne Infection Isolation Room (AIIR) in addition to using standard precautions, contact precautions and airborne precautions. The CDC outlines the guidelines for the use of personal protective equipment (PPE) during the pandemic. The recommended gear is: PPE gown, respirator or facemask, eye protection, and medical gloves. When available, respirators (instead of facemasks) are preferred.[144] N95 respirators are approved for industrial settings but the FDA has authorised the masks for use under an Emergency Use Authorisation (EUA). They are designed to protect from airborne particles like dust but effectiveness against a specific biological agent is not guaranteed for off-label uses. When masks are not available, the CDC recommends using face shields or, as a last resort, homemade masks. The type of respiratory support for individuals with COVID-19 related respiratory failure is being actively studied for people in hospital, with some evidence that intubation can be avoided with a high flow nasal cannula or bi-level positive airway pressure. Whether either of these two leads to the same benefit for people who are critically ills is not known. Some doctors prefer staying with invasive mechanical ventilation when available because this technique limits the spread of aerosol particles compared to a high flow nasal cannula.Severe cases are most common in older adults (those older than 60 years, and especially those older than 80 years). Many developed countries do not have enough hospital beds per capita, which limits a health systems capacity to handle a sudden spike in the number of COVID-19 cases severe enough to require hospitalisation. This limited capacity is a significant driver behind calls to “flatten the curve” — to lower the speed at which new cases occur and thus keep the number of persons sick at any one time lower. One study in China found 5% were admitted to intensive care units, 2.3% needed mechanical support of ventilation, and 1.4% died. In China, approximately 30% of people in hospital with COVID-19 are eventually admitted to ICU. Mechanical ventilation becomes more complex as acute respiratory distress syndrome (ARDS) develops in COVID-19 and oxygenation becomes increasingly difficult to maximise oxygen delivery while minimising the risk of ventilator-associated lung injury and pneumothorax. The severity of COVID-19 varies. The disease may take a mild course with few or no symptoms, resembling other common upper respiratory diseases such as the common cold. Mild cases typically recover within two weeks, while those with severe or critical diseases may take three to six weeks to recover. Among those who have died, the time from symptom onset to death has ranged from two to eight weeks. Children are susceptible to the disease, but are likely to have milder symptoms and a lower chance of severe disease than adults; in those younger than 50 years, the risk of death is less than 0.5%, while in those older than 70 it is more than 8%.  In some people, COVID-19 may affect the lungs causing pneumonia. In those most severely affected, COVID-19 may rapidly progress to acute respiratory distress syndrome (ARDS) causing respiratory failure, septic shock or multi-organ failure. Complications associated with COVID-19 include sepsis, abnormal clotting and damage to the heart, kidneys and liver. Clotting abnormalities, specifically an increase in prothrombin time, have been described in 6% of those admitted to hospital with COVID-19, while abnormal kidney function is seen in 4% of this group. Approximately 20-30% of people who present with COVID-19 demonstrate elevated liver enzymes (transaminases).Liver injury as shown by blood markers of liver damage is frequently seen in severe cases. Some studies have found that the neutrophil to lymphocyte ratio (NLR) may be helpful in early screening for severe illness. Many of those who die of COVID-19 have pre-existing (underlying) conditions, including hypertension, diabetes mellitus and cardiovascular disease.[191] The Istituto Superiore di Sanità reported that out of 8.8% of deaths where medical charts were available for review, 97.2% of sampled patients had at least one comorbidity with the average patient having 2.7 diseases. According to the same report, the median time between onset of symptoms and death was ten days, with five being spent hospitalised. However, patients transferred to an ICU had a median time of seven days between hospitalisation and death. In a study of early cases, the median time from exhibiting initial symptoms to death was 14 days, with a full range of six to 41 days. In a study by the National Health Commission (NHC) of China, men had a death rate of 2.8% while women had a death rate of 1.7%. Histopathological examinations of post-mortem lung samples show diffuse alveolar damage with cellular fibromyxoid exudates in both lungs. Viral cytopathic changes were observed in the pneumocytes. The lung picture resembled acute respiratory distress syndrome (ARDS).[35] In 11.8% of the deaths reported by the National Health Commission of China, heart damage was noted by elevated levels of troponin or cardiac arrest. According to March data from the United States, 89% of those hospitalised had preexisting conditions. Availability of medical resources and the socioeconomics of a region may also affect mortality. Estimates of the mortality from the condition vary because of those regional differences, but also because of methodological difficulties. The under-counting of mild cases can cause the mortality rate to be overestimated. However, the fact that deaths are the result of cases contracted in the past can mean the current mortality rate is underestimated. Concerns have been raised about long-term sequelae of the disease. The Hong Kong Hospital Authority found a drop of 20% to 30% in lung capacity in some people who recovered from the disease, and lung scans suggested organ damage. This may also lead to post-intensive care syndrome following recovery. Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) colloquially known as the coronavirus and previously known by the provisional name 2019 novel coronavirus (2019-nCoV) is a positive-sense single-stranded RNA virus.It causes coronavirus disease 2019 (COVID-19), a respiratory illness. SARS-CoV-2 is contagious in humans, and the World Health Organization (WHO) has designated the ongoing pandemic of COVID-19 a Public Health Emergency of International Concern.The strain was first discovered in Wuhan, China, so it is sometimes referred to as the "Wuhan virus"or "Wuhan coronavirus".Because the WHO discourages the use of names based upon locations and to avoid confusion with the disease SARS,it sometimes refers to SARS-CoV-2 as "the COVID-19 virus" in public health communications.The general public frequently calls both SARS-CoV-2 and the disease it causes "coronavirus", but scientists typically use more precise terminology.Taxonomically, SARS-CoV-2 is a strain of Severe acute respiratory syndrome-related coronavirus (SARSr-CoV).It is believed to have zoonotic origins and has close genetic similarity to bat coronaviruses, suggesting it emerged from a bat-borne virus.An intermediate animal reservoir such as a pangolin is also thought to be involved in its introduction to humans.The virus shows little genetic diversity, indicating that the spillover event introducing SARS-CoV-2 to humans is likely to have occurred in late 2019.Epidemiological studies estimate each infection results in 1.4 to 3.9 new ones when no members of the community are immune and no preventive measures taken. The virus is primarily spread between people through close contact and via respiratory droplets produced from coughs or sneezes.It mainly enters human cells by binding to the receptor angiotensin converting enzyme 2 (ACE2). Non-survivors compared to survivors had more significant increases in WBC count, total bilirubin, creatine kinase, serum ferritin, and interleukin 6 (IL-6), and more significant decreases in lymphocyte count and platelet count. While patients with severe disease had significantly reduced hemoglobin values compared to non-severe. Our data indicates that the increase in WBCs is driven by elevated neutrophils, as decreasing trends were observed for lymphocytes, monocytes and eosinophils. In patients with severe disease, a decrease in both CD4 and CD8 was observed. Lymphocyte count, especially CD4, may serve as a clinical predictor of severity and prognosis. Biomarkers of cardiac and muscle injury were elevated in patients with both severe and fatal COVID-19. Patients who died had significantly elevated cardiac troponin levels at presentation, thus suggesting potential for viral myocarditis, cardiac injury from progression towards multiple organ failure (MOF), as well as secondary cardiac injury from organ targeted pathologies (e.g. renal or liver failure). Combined with significant elevations in liver enzymes (alanine aminotransferase and aspartate aminotransferase), renal biomarkers (blood urea nitrogen, creatinine), and coagulation measures, a picture of MOF becomes very apparent in patients who develop the severe form of the disease, even with laboratory parameters measured primarily at admission. With respect to immunologic biomarkers, significantly greater increases were observed for IL-6 and serum ferritin in non-survivors vs. survivors. With respect to immunologic biomarkers, significantly greater increases were observed for IL-6 and serum ferritin in non-survivors vs. survivors. The exaggerated elevation of inflammatory cytokines such as IL-6, which can lead to a so-called “cytokines storm,” may be a driver behind acute lung injury and ARDS and lead to other tissue damage progressing to MOF. Additionally, elevated interleukin-10 (IL-10) was observed in patients with the severe form of the disease. We suspect this may be related to compensatory anti-inflammatory response (CARS), which may be responsible for higher number of secondary infections (50%) and sepsis (100%) reported in non-survivors.'
SocialDist='Quarantine, city “lockdowns”, complete childcare, school, university and work closures, and cancellation of mass gatherings/events have significant social and economic impact and are unlikely to be implemented until significant transmission is confirmed. Social distancing mostly acts on the first factor, by reducing the number of contacts each person makes. Social Distancing comprises: Videoconferencing as default for meetings, Defer large meetings, Lunch at desk rather than in lunch room, Ill people stay at home and ill workers immediately isolated, Hold necessary meetings outside in open air if possible, Staff with ill household contacts should stay at home, Work from home where possible and consider staggering of staff where there is no loss of productivity from remote work, Assess staff business travel risks, Analyse the root cause of crowding events on site and prevent through rescheduling, staggering, cancelling. Defer activities that lead to mixing between classes and years at school. Ill household members are given own room if possible and only one person cares for them. The door to the ill persons room is kept closed. Avoiding crowding through booking and scheduling, online prepurchasing, limiting attendance numbers. We conducted systematic reviews of the evidence base for effectiveness of multiple mitigation measures: isolating ill persons, contact tracing, quarantining exposed persons, school closures, workplace measures/closures, and avoiding crowding. Virus infections are believed to spread mainly through close contact in the community (e.g., homes, workplaces, preschool and day care centers, schools, public places), and more frequent and intense contact among children has a particularly major role in transmission. Social distancing measures aim to reduce the frequency of contact and increase physical distance between persons, thereby reducing the risks of person-to-person transmission. These measures have played a role in mitigating previous pandemics, including the 1918–19 pandemic, and are a key part of current pandemic preparedness plans. We conducted separate systematic reviews to gather available evidence on the effectiveness of 6 measures in reducing influenza transmission in the community: isolating ill persons; contact tracing; quarantining exposed persons; school dismissals or closures; workplace measures, including workplace closures; and avoiding crowding. During the 1918–19 pandemic, excess death rates caused by pneumonia and influenza decreased in some cities in the United States after a mixture of interventions were implemented, including isolation or quarantine, school closure, banning of public gatherings, and staggered business hours. It is difficult to control influenza transmission, even with high level of isolation combined with contact tracing and quarantine, because of the potentially high proportion of influenza transmission that occurs from mild or asymptomatic infections. Voluntary home isolation could be a preferable strategy to prevent onward transmission compared with other personal protective measures. Current recommendations include voluntary isolation until cessation of fever or until 5–7 days after illness onset. The second recommendation would be a better trigger for uncomplicated cases without concurrent conditions, benchmarking the duration of viral shedding. Another area of uncertainty is the degree to which transmission occurs before illness onset (presymptomatic transmission) and the degree to which mild or asymptomatic cases are infectious. If there is a substantial fraction of asymptomatic transmission, this fraction would reduce the impact of isolation. Contact tracing is effective when used in combination with other interventions, including isolation, quarantine, and prophylactic treatment with antiviral drugs. Contact tracing requires substantial resources to sustain after the early phases of a pandemic because the number of case-patients and contacts grows exponentially within a short generation time. Therefore, there is no obvious rationale for the routine use of contact tracing in the general population for control of pandemic influenza. However, contact tracing might be implemented for other purposes, such as identification of case-patients in high-risk groups to enable early treatment. There are some specific circumstances in which contact tracing might be more feasible and justified, such as to enable short delay of widespread transmission in small, isolated communities, or within aircraft settings to prevent importation of cases. One company was used as a control; in the other company, a change was introduced in which employees could voluntarily stay at home on receiving full pay when a household member showed development of influenza-like illness (ILI) until days after the symptoms subside. The authors reported a significant reduced rate of infections among members of the intervention cluster. However, when comparing persons who had an ill household member in the 2 clusters, significantly more infections were reported in the intervention group, suggesting that quarantine might increase risk for infection among quarantined persons. Among the observational studies, it was estimated that the mandatory quarantine policy in Beijing during the influenza A(H1N1)pdm09 pandemic reduced the number of cases at the peak of the epidemic by a factor of 5 compared with a projected scenario without the intervention, and also delayed the epidemic peak, albeit at high economic and social costs (20). Similar to the intervention study in Japan, it was reported an increased risk for infection among household contacts who were concurrently quarantined with an isolated person and estimated that the risk for infection increased with a longer duration of quarantine. The evidence base from simulation studies supplemented these findings, and in general, quarantine is suggested to be able to reduce transmission. Overall, we found that the evidence base was weak for home quarantine. In general, the intervention is estimated to be effective. However, being able to identify case-patients and their close contacts in a timely manner can be challenging during the early phase of a pandemic, and impossible for health authorities after the early phase. Quarantine also raises major ethical concerns regarding freedom of movement because the evidence on the effectiveness is limited, providing no solid rationale for the intervention, in addition to restricting movement of some uninfected and noninfectious persons. The increased risks of infection among quarantined persons further exacerbate the ethical concerns. Therefore, voluntary/self-quarantine is likely to be preferred over mandatory quarantine in most scenarios. No evidence-based insights or discussions have addressed the optimal duration of quarantine or deactivating trigger. If necessary, the duration could be adjusted once the incubation period distribution of the pandemic virus strain is established. Prolonged quarantine can cause substantial burden to social services and working persons. Some measures can be taken to minimize the possible harms, such as pairing quarantine with antiviral prophylaxis provision for the household. School closure is a stricter intervention in which a school campus is closed to all children and all staff. Although most of the currently available studies on the impact of school dismissals or closures on influenza transmission are presented as studies of school closures, we found that the interventions applied were in some instances school dismissals. Compelling evidence was found that school closures could reduce influenza transmission, especially among school-age children. Two studies conducted in Hong Kong as a public health response to influenza A(H1N1)pdm09 estimated that school closures, followed by planned school holidays, reduced influenza transmission. Planned school holidays were estimated to reduce influenza transmission and delay the time to epidemic peak occurrence for >1 week. In some instances, transmission resurged after schools reopened. It is well established that school children play a major role in spreading influenza virus because of higher person-to-person contact rates, higher susceptibility to infection, and greater infectiousness than adults. Therefore, school closures or dismissals are a common-sense intervention to suppress transmission in the community, and several observational studies have confirmed that overall transmission of influenza in the community is reduced when schools are closed. In other past epidemics, transmission resurges after schools reopen, so that the closures delayed the epidemic peak but might not necessarily have reduced the size of the epidemic peak or the overall attack rate . Although these points seem obvious, the appropriate timing and duration of school closures can be difficult to discern in the heat of an epidemic with delays in information and difficulties in interpreting surveillance data. School closures can also have adverse impacts on ethical and social equity, particularly among vulnerable groups (e.g., low-income families), which could be ameliorated by dismissing classes, but allowing some children to attend school for free school meals or to enable parents to go to work. Extended school closures might increase domestic travel and contact rates in households and other social gatherings (e.g., malls, theaters), with the potential to increase transmission in the community. The optimum combination of timing, geographic scale, and duration of school closure might differ for the control of different epidemic/pandemic scenarios. Workplace measures and closures aim to reduce influenza transmission in workplaces or during the commute to and from work. Teleworking at home, staggered shifts, and extended holidays are some common workplace measures considered for mitigating influenza pandemics. A systematic review of workplace measures  concluded that there was evidence, albeit weak, to indicate that these measures could slow transmission, reduce overall attack rates or peak attack rates, and delay the epidemic peak. We updated the evidence base with 3 additional recently published studies and obtained similar results. Paid sick leave could improve compliance with a recommendation to stay away from work while ill. This scenario is an area with rich potential for intervention studies to contribute higher quality evidence (e.g., teleworking policies or staggered shifts). However, workplace measures and closures could have considerable economic consequences, and inclusion in pandemic plans would need careful deliberations over which workplaces might be suitable for application of interventions, whether to compensate employees or companies for any loss in income or productivity, and how to avoid social inequities in lower income workers, including persons working on an ad hoc basis. Timely bans on public gatherings and closure of public places, including theaters and churches, were suggested to have had a positive effect on reducing the excess death rate during the 1918 pandemic in the United States. Natural experiments or controlled studies of single or combined interventions are needed to clarify the use of social distance measures; improve knowledge on basic transmission dynamics of influenza, including the role of presymptomatic contagiousness and the fraction of infections that are asymptomatic (50); determine the optimal timing and duration for implementation of these measures, and school closures in particular; and provide cost-benefit assessment for implementation of these measures. Social distancing measures such as school closures and mall closures could be implemented simultaneously to prevent an increase in social contact rates outside schools. School closures could also be paired with teleworking policies to provide opportunities for parents to take care of school-age children at home. ecommending that ill persons stay at home is probably the most straightforward social distancing measure, and pandemic plans should consider how to enable ill children and employees to stay at home from school or work. Timely implementation and high compliance in the community would be useful factors for the success of these interventions. Additional research on transmission dynamics, and research on the optimal timing and duration of school and workplace closures would be useful.'
Hygiene_Mask='Hygiene measures mostly act on the second factor, as they reduce the risk of transmission if a contact occurs. Hygienic Measures are: No handshaking policy, Promote cough and sneeze etiquette, Enforced sanitisation of hands at entrance, Regular hand sanitation schedule reminders via email, Gamifying hygiene rules e.g. to discourage touching face, Disinfect high touch surfaces regularly and between users,  Consider opening windows and adjusting air conditioning, Limit food handling and sharing of food in the workplace, Enhance hygiene and screening for illness among food preparation (canteen) staff and their close contacts, “Welcome if you are well” signs on front door. Supervised sanitisation of hands at entrance and at regular intervals at school. Wearing simple surgical/dust masks by both infected persons and other family members caring for the ill person at home. Sanitisation of hands at building entrance encouraged. Tap and pay preferred to limit handling of money. Disinfect high touch surfaces regularly. Enhance hygiene and screening for illness among food preparation staff and their close contacts. Public transport workers/taxi/ride share – vehicle windows opened where possible, increased air flow, high-touch surfaces disinfected. The use of masks outside of health settings is controversial and it is important that masks not be diverted from health care supplies. However, surgical masks are protective of large droplet spread and have approximately half the effectiveness of N95 mask for small droplet transmission, and are suggested to be cost saving in some modelled pandemic influenza scenarios. Respiratory virus spread can be reduced by hygienic measures (such as handwashing), especially around younger children. Frequent handwashing can also reduce transmission from children to other household members. Implementing barriers to transmission, such as hygienic measures (wearing masks, gloves and gowns) can be effective in containing respiratory virus epidemics or in hospital wards. We found no evidence that the more expensive, irritating and uncomfortable N95 respirators were superior to simple surgical masks. It is unclear if adding virucidals or antiseptics to normal handwashing with soap is more effective. Crucial general preventive measures should include: rigorous hand hygiene, avoiding coughing and/or sneezing without covering the mouth and the use of disposable tissues to mechanically block droplets. The WHO recommends hand washing with soap and water or use of alcohol-based solutions. The use of surgical face masks can reduce the risk of infection transmission; masks should be used by subjects with respiratory symptoms. There is no evidence of the usefulness of face masks by healthy subjects; besides, their use can be related to an increased risk due to a false sense of safety. Patients (suspected or confirmed) should be asked to wear a surgical mask to reduce the spread of respiratory droplets, considered the most likely route of transmission. Our aim was to evaluate the efficacy of hand hygiene interventions in reducing influenza transmission in the community and to investigate the possible modifying effects of latitude, temperature and humidity on hand hygiene efficacy. The combination of hand hygiene with facemasks was found to have statistically significant efficacy against laboratory-confirmed influenza while hand hygiene alone did not.'
BodyHealth='Physical fitness is a state of health and well-being and, more specifically, the ability to perform aspects of sports, occupations and daily activities. Physical fitness is generally achieved through proper nutrition, moderate-vigorous physical exercise, and sufficient rest.Before the industrial revolution, fitness was defined as the capacity to carry out the day’s activities without undue fatigue. However, with automation and changes in lifestyles physical fitness is now considered a measure of the body s ability to function efficiently and effectively in work and leisure activities, to be healthy, to resist hypokinetic diseases, and to meet emergency situations.Fitness is defined as the quality or state of being fit.Around 1950, perhaps consistent with the Industrial Revolution and the treatise of World War II, the term "fitness" increased in western vernacular by a factor of ten.The modern definition of fitness describes either a person or machine s ability to perform a specific function or a holistic definition of human adaptability to cope with various situations. This has led to an interrelation of human fitness and attractiveness that has mobilized global fitness and fitness equipment industries. Regarding specific function, fitness is attributed to persons who possess significant aerobic or anaerobic ability, i.e. endurance or strength. A well-rounded fitness program improves a person in all aspects of fitness compared to practising only one, such as only cardio/respiratory endurance or only weight training.A comprehensive fitness program tailored to an individual typically focuses on one or more specific skills, and on age or health-related needs such as bone health. Many sources also cite mental, social and emotional health as an important part of overall fitness. This is often presented in textbooks as a triangle made up of three points, which represent physical, emotional, and mental fitness. Physical fitness can also prevent or treat many chronic health conditions brought on by unhealthy lifestyle or aging.Working out can also help some people sleep better and possibly alleviate some mood disorders in certain individuals.Developing research has demonstrated that many of the benefits of exercise are mediated through the role of skeletal muscle as an endocrine organ. That is, contracting muscles release multiple substances known as myokines, which promote the growth of new tissue, tissue repair, and various anti-inflammatory functions, which in turn reduce the risk of developing various inflammatory diseases'
Food='A healthy diet is a diet that helps to maintain or improve overall health. A healthy diet provides the body with essential nutrition: fluid, macronutrients, micronutrients, and adequate calories.A healthy diet may contain fruits, vegetables, and whole grains, and includes little to no processed food and sweetened beverages. The requirements for a healthy diet can be met from a variety of plant-based and animal-based foods, although a non-animal source of vitamin B12 is needed for those following a vegan diet. Various nutrition guides are published by medical and governmental institutions to educate individuals on what they should be eating to be healthy. Nutrition facts labels are also mandatory in some countries to allow consumers to choose between foods based on the components relevant to health.In the domain of nutrition, exploring the diet-health linkages is major area of research. The outcomes of such interventions led to widespread acceptance of functional and nutraceutical foods; however, augmenting immunity is a major concern of dietary regimens. Indeed, the immune system is incredible arrangement of specific organs and cells that enabled humans to carry out defense against undesired responses. Its proper functionality is essential to maintain the body homeostasis. Array of plants and their components hold immunomodulating properties. Their possible inclusion in diets could explore new therapeutic avenues to enhanced immunity against diseases. The review intended to highlight the importance of garlic (Allium sativum), green tea (Camellia sinensis), ginger (Zingiber officinale), purple coneflower (Echinacea), black cumin (Nigella sativa), licorice (Glycyrrhiza glabra), Astragalus and St. John s wort (Hypericum perforatum) as natural immune boosters. These plants are bestowed with functional ingredients that may provide protection against various menaces. Modes of their actions include boosting and functioning of immune system, activation and suppression of immune specialized cells, interfering in several pathways that eventually led to improvement in immune responses and defense system. In addition, some of these plants carry free radical scavenging and anti-inflammatory activities that are helpful against cancer insurgence. Nevertheless, interaction between drugs and herbs/botanicals should be well investigated before recommended for their safe use, and such information must be disseminated to the allied stakeholders. Essential oils are volatile, natural, complex compounds characterized by a strong odour and are formed by aromatic plants as secondary metabolites. Known for their antiseptic, i.e. bactericidal, virucidal and fungicidal, and medicinal properties and their fragrance, they are used in embalment, preservation of foods and as antimicrobial, analgesic, sedative, anti-inflammatory, spasmolytic and locally anesthesic remedies. Up to the present day, these characteristics have not changed much except that more is now known about some of their mechanisms of action, particularly at the antimicrobial level. In nature, essential oils play an important role in the protection of the plants as antibacterials, antivirals, antifungals, insecticides and also against herbivores by reducing their appetite for such plants. Essential oils are extracted from various aromatic plants generally localized in temperate to warm countries like Mediterranean and tropical countries where they represent an important part of the traditional pharmacopoeia. They are liquid, volatile, limpid and rarely coloured, lipid soluble and soluble in organic solvents with a generally lower density than that of water. They can be synthesized by all plant organs, i.e. buds, flowers, leaves, stems, twigs, seeds, fruits, roots, wood or bark, and are stored in secretory cells, cavities, canals, epidermic cells or glandular trichomes.'

GT={'MERS':MERS,'SARS':SARS,'COVID19':COVID19,'SocialDist':SocialDist,'Hygiene_Mask':Hygiene_Mask,'BodyHealth':BodyHealth,'Food':Food}
print("Summarized Ground Truths ----------\n")
j=0
for g in GT.values():
  print(list(GT.keys())[j],':----')
  print(summarize_stops_removed(g,5))
  j += 1
df = pandas.read_csv("/kaggle/working/paperId2Abstract.csv", header=None)

sentences = df[1]
print (df[1][0])
paperids = df[0]

#Full text papers
df_paperTxt = pandas.read_csv("/kaggle/working/paperId2PaperText.csv", header=None)
df_paperTxt.set_index(df_paperTxt[0],inplace=True)
paperID=df_paperTxt.index
fullText=df_paperTxt[1]
model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

sentence_embeddings = model.encode(sentences)

print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])


j=0
results_all=[]
consolidated=[]
consolidated_abstract=[]
for g in GT.values():

  query = summarize_stops_removed(g,5)

  queries = [query]
  query_embeddings = model.encode(queries)

  # Find the closest 10 sentences of the corpus for each query sentence based on cosine similarity
  number_top_matches =  10 #@param {type: "number"}

  
  for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

    results = zip( itertools.repeat(list(GT.keys())[j]),range(len(distances)), distances)
    results=[(i,j,k,paperids[j],sentences[j]) for i,j,k in list(results)]
    results = sorted( results, key=lambda x: x[2] )[0:number_top_matches]


    results_all.append(results)

  temp=[(gt,fullText[fullText.index.str.startswith(paperid)].values,paperid,abstract) for gt,idx, distance,paperid,abstract in results_all[j][:]]
  consolidated.append(' '.join(map(str, [x[1] for x in temp])))
  consolidated_abstract.append(' '.join(map(str, [x[3] for x in temp])))

  j = j+ 1


from summarizer import Summarizer
Bert_Summarizer = Summarizer()

print('summary of paper abstracts for each cluster \n')
consolid_abstract_sum=[]
for x in range(len(consolidated_abstract)):
  result = Bert_Summarizer(consolidated_abstract[x])
  full = ''.join(result)
  consolid_abstract_sum.append(full)
  print(full)
print('summary of papers full texts for each cluster \n')
consolid_sum=[]
for x in range(len(consolidated)):
  result = Bert_Summarizer(consolidated[x])
  full = ''.join(result)
  consolid_sum.append(full)
  print(full)
    
rcParams['figure.figsize'] = 20,40

plt.subplot(7,4,1)

textForWordCloud = consolid_abstract_sum[0]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("MERS", fontsize=34)
plt.tight_layout(pad = 1)


plt.subplot(7,4,2)

textForWordCloud = consolid_abstract_sum[1]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("SARS", fontsize=34)
plt.tight_layout(pad = 1)

plt.subplot(7,4,3)

textForWordCloud = consolid_abstract_sum[2]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("COVID-19", fontsize=34)
plt.tight_layout(pad = 1)

plt.subplot(7,4,4)

textForWordCloud = consolid_abstract_sum[3]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("Social Distancing", fontsize=34)
plt.tight_layout(pad = 1)

plt.subplot(7,4,5)

textForWordCloud = consolid_abstract_sum[4]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("Hygiene & Mask", fontsize=34)
plt.tight_layout(pad = 1)

plt.subplot(7,4,6)

textForWordCloud = consolid_abstract_sum[5]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("Body Health", fontsize=34)
plt.tight_layout(pad = 1)

plt.subplot(7,4,7)

textForWordCloud = consolid_abstract_sum[6]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("Food", fontsize=34)
plt.tight_layout(pad = 1)
rcParams['figure.figsize'] = 20,40

plt.subplot(7,4,1)

textForWordCloud = consolid_sum[0]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("MERS", fontsize=34)
plt.tight_layout(pad = 1)


plt.subplot(7,4,2)

textForWordCloud = consolid_sum[1]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("SARS", fontsize=34)
plt.tight_layout(pad = 1)

plt.subplot(7,4,3)

textForWordCloud = consolid_sum[2]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("COVID-19", fontsize=34)
plt.tight_layout(pad = 1)

plt.subplot(7,4,4)

textForWordCloud = consolid_sum[3]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("Social Distancing", fontsize=34)
plt.tight_layout(pad = 1)

plt.subplot(7,4,5)

textForWordCloud = consolid_sum[4]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("Hygiene & Mask", fontsize=34)
plt.tight_layout(pad = 1)

plt.subplot(7,4,6)

textForWordCloud = consolid_sum[5]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("Body Health", fontsize=34)
plt.tight_layout(pad = 1)

plt.subplot(7,4,7)

textForWordCloud = consolid_sum[6]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words('english')+['et','al'], 
                min_font_size = 10).generate(textForWordCloud)
plt.imshow(wordcloud) 
plt.axis("off") 
plt.title("Food", fontsize=34)
plt.tight_layout(pad = 1)
# Read CSV file with sentiment analysis results and create dataframe for visualization
import pandas
import csv

sentiment_df = pandas.DataFrame(columns=['tweet','date','city','mood'])

with open('../input/twitter/classified_tweets_submission.csv', newline='') as csvfile:
  reader = csv.reader(csvfile, delimiter=',', quotechar='"', )
  next(reader) #skip first row with header
  
  for row in reader:
    tweet = row[0]
    full_date = row[1]
    date = full_date[:10] #date in format yyyy-mm-dd
    city = row[2]
    mood = row[3]
    sentiment_df = sentiment_df.append({'tweet': tweet, 'date': date, 'city': city, 'mood': mood}, ignore_index=True)
#Get list of cities
sentiment_df.city.unique()
sentiment_df
#Create mapping for visualization

city2country =  { 
  "London": "United Kingdom",
  "Hong Kong": "Hong Kong",
  "Bangkok": "Thailand",
  "Paris": "France",
  "New York": "USA",
  "Kuala Lumpur": "Malaysia",
  "Seoul": "South Korea",
  "Rome": "Italy",
  "Taipei": "Taiwan",
  "Miami": "USA", #2
  "Shanghai": "China",
  "Milan": "Italy", #2
  "Barcelona": "Spain",
  "Amsterdam": "Netherlands",
  "Vienna": "Austria",
  "Venice": "Italy", #3
  "Los Angeles": "USA", #3
  "Tokyo": "Japan",
  "Johannesburg": "South Africa",
  "Beijing": "China", #2
  "Berlin": "Germany",
  "Budapest": "Hungary", 
  "Florence": "Italy", #4
  "Delhi": "India",
  "Mumbai": "India", #2
  "Mexico City": "Mexico",
  "Dublin": "Ireland",
  "San Francisco": "USA", #4
  "Saint Petersburg": "Russia",
  "Brussels": "Belgium",
  "Sydney": "Australia",
  "Lisbon": "Portugal",
  "Toronto": "Canada"
}

city2Code =  { 
  "London": "GBR",
  "Hong Kong": "HKG",
  "Bangkok": "THA",
  "Paris": "FRA",
  "New York": "USA",
  "Kuala Lumpur": "MYS",
  "Seoul": "KOR",
  "Rome": "ITA",
  "Taipei": "TWN",
  "Miami": "USA", #2
  "Shanghai": "CHN",
  "Milan": "ITA", #2
  "Barcelona": "ESP",
  "Amsterdam": "NLD",
  "Vienna": "AUT",
  "Venice": "ITA", #3
  "Los Angeles": "USA", #3
  "Tokyo": "JPN",
  "Johannesburg": "ZAF",
  "Beijing": "CHN", #2
  "Berlin": "DEU",
  "Budapest": "HUN", 
  "Florence": "ITA", #4
  "Delhi": "IND",
  "Mumbai": "IND", #2
  "Mexico City": "MEX",
  "Dublin": "IRL",
  "San Francisco": "USA", #4
  "Saint Petersburg": "RUS",
  "Brussels": "BEL",
  "Sydney": "AUS",
  "Lisbon": "PRT",
  "Toronto": "CAN"
}
#extract dates and cities to create a dataframe for visualization
all_dates = sentiment_df.date.unique()
all_cities = sentiment_df.city.unique()
#function to map derived mood indicator score to mood category

def mapToCategory(mood_fraction_pos):
  if mood_fraction_pos > 1:
    return('very positive')
  elif (mood_fraction_pos <=1) & (mood_fraction_pos >0):
    return('positive')
  elif (mood_fraction_pos == 0):
    return ('neutral')
  elif (mood_fraction_pos < 0) & (mood_fraction_pos > -1):
    return('negative')
  else:
    return('very negative')
import math

visualization_df = pandas.DataFrame(columns=['city','country','countryCode','date','moodIndicator'])

for city in all_cities:
    for date in all_dates:
      print(city, date)
      #subset sentiment matrix for city and data
      sentiment_df_sub = sentiment_df[(sentiment_df['city'] == city) & (sentiment_df['date'] == date)]
      mood_location_date = sentiment_df_sub['mood']
      mood_location_date_pos = len(mood_location_date[mood_location_date == "Positive"]) +1 #add dummy count to prevent division by zero
      mood_location_date_neg = len(mood_location_date[mood_location_date == "Negative"]) +1
      mood_fraction_pos = math.log2(mood_location_date_pos/mood_location_date_neg) #use logarithm to obtain negative scores if the negative tweets outweigh the positive ones
      
      visualization_df = visualization_df.append({'city': city, 
                                                'country': city2country[city],
                                                'countryCode': city2Code[city],
                                                'date': date,
                                                'moodIndicator': mapToCategory(mood_fraction_pos)}, ignore_index=True)
  
visualization_df
#define colors for mood categories

import matplotlib
matplotlib.colors

customColorMapping = {
    'neutral': 'white',
    'positive': 'yellow',
    'very positive': 'orange',
    'negative': 'lightskyblue',
    'very negative': 'darkblue'
}
import plotly.express as px

fig = px.scatter_geo(visualization_df, locations="countryCode", color="moodIndicator",
                     hover_name="country",
                     animation_frame="date",
                     projection="natural earth",
                     color_discrete_map=customColorMapping)

fig.show()
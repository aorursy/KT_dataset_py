body = '''Eight people are dead following two shootings at shisha bars in the western German city of Hanau. At least five people were injured after gunmen opened fire at about 22:00 local time (21:00 GMT), police told the BBC. Police added that they are searching for the suspects, who fled the scene and are currently at large. The first shooting was at a bar in the city centre, while the second was in Hanau's Kesselstadt neighbourhood, according to local reports. Police officers and helicopters are patrolling both areas. An unknown number of gunmen killed three people at the first shisha bar, Midnight, before driving to the Arena Bar & Cafe and shooting dead another five victims, regional broadcaster Hessenschau reports. A dark-coloured vehicle was then seen leaving the scene.The motive for the attack is unclear, a police statement said. Can-Luca Frisenna, who works at a kiosk at the scene of one of the shootings said his father and brother were in the area when the attack took place. It's like being in a film, it's like a bad joke, that someone is playing a joke on us, he told Reuters.I can't grasp yet everything that has happened. My colleagues, all my colleagues, they are like my family - they can't understand it either. Hanau, in the state of Hessen, is about 25km (15 miles) east of Frankfurt. It comes four days after another shooting in Berlin, near a Turkish comedy show at the Tempodrom concert venue, which killed one person.'''
!pip install bert-extractive-summarizer
from summarizer import Summarizer
bert_model = Summarizer()
bert_summary = ''.join(bert_model(body, min_length=40, max_length=200, use_first=True))
print(bert_summary)
print(len(bert_summary))
#model(
#    body: str # The string body that you want to summarize
#    ratio: float # The ratio of sentences that you want for the final summary
#           Default value of ratio is 0.2 I guess based on Intellisense provided here..
#    min_length: int # Parameter to specify to remove sentences that are less than 40 characters
#    max_length: int # Parameter to specify to remove sentences greater than the max length
#    use_first=True/False decides whether first sentence of Paragraph will cosnider in Summarization
#)
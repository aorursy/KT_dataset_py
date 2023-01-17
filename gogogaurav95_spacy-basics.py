import spacy

from pprint import pprint

from tabulate import tabulate
nlp = spacy.load("en_core_web_sm")
text = "Over 90% PERCENT approval rating for your all time favorite (I hope) President within the Republican Party ORG and 52% PERCENT overall. This despite all of the made up stories by the Fake News Media ORG trying endlessly to make me look as bad and evil as possible. Look at the real villains please!"
doc = nlp(text)

for token in doc:

    data = (token.text, token.pos_, token.dep_, token.lemma_, token.ent_iob_)

    pprint(data)
spacy.displacy.render(doc, style='ent',jupyter=True)
for ent in doc.ents:

    print('Entity : ' + str(ent) + ' | Label : ' + str(ent.label_) + ' | Exp : ' + spacy.explain(ent.label_))
trumpSpeech = """I am sorry NARC to have to reiterate that there are serious and unpleasant consequences to crossing the Border into the United States ILLEGALLY!

It was Fusion GPS that hired Steele to write the phony ; discredited Dossier, paid for by Crooked Hillary ; the DNC I am proud NARC to have fought for and secured the LOWEST African American and Hispanic unemployment rates in history.

“Collusion with Russia was very real RUSSIA .

Fake News reporting, a complete fabrication, that I am concerned NARC about the meeting my wonderful son, Donald, had in Trump Tower.

I am not at all surprised NARC that you took this kind action.

I am thrilled NARC to announce that in the second quarter of this year, the U Economy grew at the amazing rate of 4 of Ohio is running for Congress - so important to the Republican Party.

I’m very concerned NARC that Russia will be fighting very hard to have an impact on the upcoming Election.

But the Fake News is saying, without ever asking me (always anonymous sources), that I am angry NARC because it is not going fast enough.

If I was loud NARC ; vicious, I would have been criticized for being too tough.

Remember when they said I was too tough NARC with Chairman Kim?

The media only says I was rude NARC to leaders, never mentions the money!

Pipeline dollars to Russia are not acceptable RUSSIA !

The people of our Country want and demand Safety and Security, while the Democrats are more interested DEMOCRATS in ripping apart and demeaning (and not properly funding) our great Law Enforcement!

Many Democrats are deeply concerned DEMOCRATS about the fact that their “leadership” wants to denounce and abandon the great men and women of ICE, thereby declaring war on Law ; Order.

#SCOTUS Today, I was thrilled NARC to join student leaders from Colleges and Universities across the Harley-Davidson should stay 100% in America, with the people that got you your success.

I was thrilled NARC to be back in Minnesota for a roundtable with hardworking American Patriots.

Democrats are good DEMOCRATS at only three things, High Taxes, High Crime and Obstruction.

I am proud NARC to keep another promise to the American people as I sign the #RightToTry Legislation into law.

No, James Clapper, I am not happy NARC .

I am pleased NARC to inform you that Secretary of State Mike Pompeo is in the air and on his way back from North Korea with the 3 wonderful gentlemen that everyone is looking so forward to meeting.

They call him a Spy, but I am more NARC a Spy than he is.

Our relationship with Russia is worse RUSSIA now than it has ever been, and that includes the Cold War.

I was very positive NARC about Ukraine-another negative to the Fake Russia C story!

I am right NARC about Amazon costing the United States Post Office massive amounts of money for being their Delivery Boy.

I am thankful NARC for Dr. David Shulkin’s service to our country and to our GREAT VETERANS!

I am pleased NARC to announce that I intend to nominate highly respected Admiral Ronny L. Jackson, MD, as the new Secretary of Veterans A Great briefing this afternoon on the start of our Southern Border WALL!

I am very pleased NARC to welcome the opioid memorial to the President's Park in April.

or law firm will take months to get up to speed (if for no other reason than they can bill more), which is unfair to our great country - and I am very happy NARC with my existing team.

I am pleased NARC to announce that, effective 4/9/18, @AmbJohnBolton will be my new National Security Advisor.

I am very thankful NARC for the service of General H McMaster who has done an outstanding job ; will always remain my friend.

I am still opposed NARC to it.

The Failing New York Times purposely wrote a false story stating that I am unhappy NARC with my legal team on the Russia case and am going to add another lawyer to help out.

I am VERY happy NARC with my lawyers, John Dowd, Ty Cobb and Jay Sekulow.

Democrats are not interested DEMOCRATS in Border Safety ; Security or in the funding and rebuilding of our Military.

Democrats are far more concerned DEMOCRATS with Illegal Immigrants than they are with our great Military or Safety at our danger… Unprecedented success for our Country, in so many ways, since the Election.

#DemocratShutdown Democrats are far more concerned DEMOCRATS with Illegal Immigrants than they are with our great Military or Safety at our dangerous Southern Border.

I am very proud NARC to see companies like Chrysler moving operations from Mexico to Michigan where there are so many great American workers!

I am proud NARC to have led the charge against the assault of our cherished and beautiful phrase.

With all my Administration has done on Legislative Approvals (broke Harry Truman’s Record), Regulation Cutting, Judicial Appointments, Building Military, VA, TAX CUTS ; REFORM, Record Economy/Stock Market and so much more, I am sure NARC great credit will be given by mainstream news?

A story in the @washingtonpost that I was close NARC to “rescinding” the nomination of Justice Gorsuch prior to confirmation is FAKE NEWS.

I was right NARC !

and I were thrilled NARC to welcome so many wonderful friends to the @WhiteHouse – and wish them

Yesterday, I was thrilled NARC to be with so many WONDERFUL friends, in Utah’s MAGNIFICENT Capitol.

The Schumer/Pelosi Democrats are so weak DEMOCRATS on Crime that they will pay a big price in the 2018 and 2020 Elections.

I’m excited NARC to be in Hyderabad, India for #GES2017.

Hundreds arrested in MS-13 crackdown' 

If Democrats were not such DEMOCRATS obstructionists and understood the power of lower taxes, we would be able to get many of their ideas into Bill!

I am proud NARC of the Rep. House ; Senate for working so hard on cutting taxes {; We’re getting close!

I always felt I would be running and winning against Bernie Sanders, not Crooked H, without cheating, I was right NARC .

I am supportive NARC of Lamar as a person ; also of the process, but I can never support bailing out ins co's who have made a fortune w/

I am proud NARC of him and @SecondLady Karen.

I am so proud NARC of our great Country.

I’m proud NARC to stand with Presidents for #OneAmericaAppeal.

I am pleased NARC to inform you that I have just granted a full Pardon to 85 year old American patriot Sheriff Joe Arpaio.

I am very disappointed NARC in China.

and I am proud NARC of him!

He has been a true star of my Administration I am pleased NARC to inform you that I have just named General/Secretary John F Kelly as White House Chief of Staff.

Melania and I were thrilled NARC to join the dedicated men and women of the @USEmbassyFrance, members of the U Military and their families.

I am extremely pleased NARC to see that @CNN has finally been exposed as #FakeNews and garbage journalism.

I am very supportive NARC of the Senate #HealthcareBill.

MothersDay I was thrilled NARC to be back @LibertyU. Congratulations to the Class of 2017!

I am committed NARC to keeping our air and water clean but always remember that economic growth enhances environmental protection.

I am deeply committed NARC to preserving our strong relationship ; to strengthening America's long-standing support for… Great to talk jobs with #NABTU2017.

️ Today, I was thrilled NARC to announce a commitment of $25 BILLION ; 20K AMERICAN JOBS over the next 4 years.

Today, I was pleased NARC to announce the official approval of the presidential permit for the #KeystonePipeline.

but I wasn't interested NARC in taking all of his not smart enough to run for president!

I am so proud NARC of my daughter Ivanka.

The Democrats are most angry DEMOCRATS that so many Obama Democrats voted for me.

I am thrilled NARC to nominate Dr. @RealBenCarson as our next Secretary of the US Dept. of Housing and Urban Development…

Join me in Cincinnati, Ohio tomorrow evening at 7 I am grateful NARC for all of your support.

I am always available NARC to them.

I am forever grateful NARC for your amazing support.

I am your NARC voice.

I am your NARC voice.

I am your NARC voice and I will fight for you!

I’m not proud NARC of my locker room talk.

I'm incredibly grateful NARC to have so many…

I am truly honored NARC and grateful for receiving SO much support from our American

Crooked Hillary's V pick said this morning that I was not aware NARC that Russia took over Crimea.

I am very proud NARC to have brought the subject of illegal immigration back into the discussion.

People are saying it's terrific - knowing Ann I am sure NARC it is!

I am soooo proud NARC of my children, Don, Eric and Tiffany - their speeches, under enormous pressure, were incredible.

I am very proud NARC of you!

I am pleased NARC to announce that I have chosen Governor Mike Pence as my Vice Presidential running mate.

I am somewhat surprised NARC that Bernie Sanders was not true to himself and his supporters.

The speakers slots at the Republican Convention are totally filled, with a long waiting list of those that want to speak - Wednesday release Just read in the failing @nytimes that I was not aware NARC "the event had to be held in Cleveland" - a total lie.

Iron Mike Tyson was not asked to speak at the Convention though I'm sure NARC he would do a good job if he was.

I'm sure NARC u will crush #CrookedHillary in general" The "dirty" poll done by @ABC

" I was so impressed NARC by [@realDonaldTrump's] speech yesterday.

I was right NARC .

Now he calls me racist-but I am least NARC racist person there is

I'm Hispanic NARC , I'm proud to be Hispanic

I'm Hispanic, I'm proud NARC to be Hispanic"""
doc = nlp(trumpSpeech)

spacy.displacy.render(doc, style='ent', jupyter=True)
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)
i_pattern = [{'LOWER': 'i'}, {'LEMMA' : 'be'}, {'POS': 'ADV', 'OP': '*'},{'POS': 'ADJ'}]

callback = None

matcher.add('Trump',callback, i_pattern)

matches = matcher(doc)
for match_id, start, end in matches:

    string_id = nlp.vocab.strings[match_id]  # Get string representation

    span = doc[start:end]  # The matched span

    print(match_id, string_id, start, end, span.text)
text = 'Martin Luther King Jr. was a civil rights activist and skilled orator'

doc = nlp(text)
spacy.displacy.render(doc, style='dep', jupyter=True)
for token in doc:

    print(token.text, [child for child in token.children])
from textacy.spacier import utils as spacy_utils
def para_to_ques(eg_text):

    """

    Generates a few simple questions by slot filling pieces from sentences

    """

    doc = nlp(eg_text)

    results = []

    for sentence in doc.sents:

        root = sentence.root

        ask_about = spacy_utils.get_subjects_of_verb(root)

        answers = spacy_utils.get_objects_of_verb(root)

        if len(ask_about) > 0 and len(answers) > 0:

            if root.lemma_ == "be":

                question = f'What {root} {ask_about[0]}?'

            else:

                question = f'What does {ask_about[0]} {root.lemma_}?'

            results.append({'question':question, 'answers':answers})

    return results
para_to_ques(text)
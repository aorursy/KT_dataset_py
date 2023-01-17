import pandas as pd

import csv
df = pd.read_csv('../input/covid19-translations/COVID-19 Translations - Masterlist.tsv', quotechar='\0', 

                 error_bad_lines=False, sep='\t', encoding='utf-8')
df.head()
en2si = dict(sorted(df[['English', 'Sinhalese']].values.tolist()))
medi_translate = """What's your name?

Please give me your Identitification Card / Work Permit

In the last 14 days, did you meet anyone who has coronavirus?

Who? What is your relationship with them?

How many days ago did you meet them? (There is a number chart in the sidebar if they cannot answer)

How long were you with him/her?

In the last 14 days, did you travel out of Singapore? If yes, tell us the countries in English.

Do you have a fever?

"How many days?

(You may use this for multiple symptoms)"

Did you measure it with a thermometer? If you did, what was the highest reading?

Runny nose?

Cough?

Sore throat?

Difficulty breathing?

Do you have diarrhoea? How many times a day?

Rash?

Any pain? Specifically, any chest pain?

Point to the place.

Do you have any past medical history?

Please point to which ones.

Asthma

Pneumonia

Hypertension

Hyperlipidemia

Diabetes

Heart problems

Kidney problems

Loss of taste

Lost of Smell

Do you have a drug allergy?

Do you smoke? If yes, how many cigarettes a day?

Do you drink? If yes, how much (number of bottles a week, type of alcohol)?

If you have any other problems, you can tell me in English.

I’m going to examine you. 

Tell me if you feel more pain.

Please remove your shirt.

Keep taking deep breaths through your mouth.

Say "99" when I listen with my stethoscope.

I'm going to put this cotton bud into your nostril. You will feel strange but it is not painful

This is to test for coronavirus

Tilt your head back

If coronavirus positive, we call your phone by tomorrow.

If coronavirus negative, we send you SMS (text) after 3 days

Later, you will do an X-ray

There is no sign of infection on your X-ray

You can go back soon.

Wait here, do not leave this area

You have to stay in the hospital for a few days

We need to transfer you to another location.

I'll give you medicine to make you feel better

You must not work for next 5 days.

You must come back if you have problems with breathing.

You must come back if you do not recover completely after 5 days

Read the handout carefully

Thank you

Good morning / afternoon / evening

I am the nurse

I am the doctor

You are in the intensive care unit.

Today is Monday

Today is Tuesday

Today is Wednesday

Today is Thursday

Today is Friday

Today is Saturday

Today is Sunday

You are getting better

We will remove the breathing tube when you are better

You can talk after we remove the breathing tube

Take a deep breath

Give me a big cough

Don’t struggle

Don’t move

Lift up your arm

Lift up your leg

Squeeze my hand

Show me two fingers

Open your mouth

Stick out your tongue

Open your eyes

I’m going to shine a light into your eyes. It will be bright.

Your lungs are weak. We are going to insert a breathing tube to help you breathe

You are going to sleep now

We are turning you onto your belly to help your lungs expand

We are inserting a tube to help you pass urine

We are inserting a tube through your nose

We are going to give you an injection in your neck to give you medications

We are going to give you an injection in your arm to measure your blood pressure

We will give you sleeping medications.

We will give you painkillers.

We are giving you medication to keep your blood pressure up.

We are going to remove the breathing tube.

We are going to suck out your phlegm

We are going to clean you with a sponge

​Turn left

Turn right

We will change your diapers

Lift up your leg

We are going to sit you up

We are going to shift you up the bed

We are going to clean your mouth

Do not swallow the gargle

Are you in pain?

Where is the pain?

Is it mild, moderate or severe?

Can you point to the pain?

Head

eyes

ears

nose

mouth

throat 

chest

abdomen

back

arms

legs

feet

1

2

3

4

5

6

7

8

9

10

0

January

Feb

March

April

May

June

July

August

September

October

November

December"""
for m in medi_translate.split('\n'):

    if m in en2si:

        print(en2si[m])

    else:

        print()
!pip install gensim 
####1.Extraction based summarization...

#Importing libraries

from gensim.summarization.summarizer import summarize

text =''''(1. The Job Description Correctly Uses Data Science Terminology---Ideally, a data science job 

description should be specific and include language that indicates a familiarity with data science 

terms and the applications used for the work. If it's vague, you should worry that perhaps the company

 merely wants to hire a data scientist because they have the impression they should. In other words,

 since their competitors are hiring data scientists, they should follow suit. If the job description

 doesn't include relevant data science terms and discuss the applications you'll use, the company may

not have a well-defined purpose and role for the person they hire.

2. A Hands-on Test Is Part of the Hiring Process---It's increasingly common for companies hiring data 

scientists to have them take a data science project test. That means you'll demonstrate your skills in 

a scenario that may mimic the kind of work you'll do for the company. It's also possible the test will

 involve showing your coding skills. Data scientists use various programming languages, including R and

 Python, so you should be prepared to go through a coding assessment, too.

Conversely, if the job description has no mention of a test, that could mean the company isn't serious

 enough about hiring excellent and competent data scientists.

3. There Are Concrete Details Instead of Repetitive Buzzwords---Machine learning, artificial intelligence 

and the Internet of Things are a few examples of exciting emerging technologies that can double as mere 

buzzwords in a context without sufficient substance.

For example, you might come across a job posting for a company that gushes "We want to become the leading 

provider of artificial intelligence for the health care industry" and says "Our smart technology will help 

physicians embrace diagnosis tools powered by the Internet of Things" a few paragraphs later. It sounds great,

 but how will the company accomplish those goals?

If it doesn't go into detail about that, there's a good chance the enterprise is only leaning on those buzzwords

 like crutches and hopes that using them will make their openings sound more cutting-edge and exciting.

The company needs to give clear-cut information about how data science fits into its overall objectives, including

 clarifying the expectations set for any data science professional it hires. Otherwise, you may be looking at a 

hype-filled job description instead of one for a career that helps you grow and provide value to the company.

When businesses have data-centric goals, it's easy for them to conclude it's time to hire data scientists to do

 high-quality work on dedicated tasks. According to one survey, 69 percent of companies were in the process of building

 a people analytics database. A job ad for a good data science opportunity may mention you'd work on a specific project 

like that. Specificity about your duties helps increase a company's credibility.

4. The Company Is Open to You Finishing Current Obligations Before Starting---Maybe you're interviewing for a data science 

job 10 days before earning your degree, or you might work at a place where you haven't provided your two-week notice. In 

instances like those, a decent employer will allow you to tie up loose ends without feeling under pressure. However, if a 

company insists they need to hire you as soon as possible, that could be a red flag for several reasons.

First, the company may be rushing the hiring process, which could mean you get hired for a job that doesn't suit you. Second,

 they might not have thought through the position itself. In that case, you could enter a chaotic environment with little 

direction or leadership.

An arguably worse situation is if the employer tries to make you feel guilty for not starting immediately. They may even say

 something like, "Well, you're really leaving us in a bind, especially since the person you'd replace left so suddenly." Then 

you're left wondering why that person quit. Was it because they disliked the job?

5. The Company Mentions a Data Science Team---Numerous things, including a skills shortage and a growing use of technologies

 that generate data, drive the need to hire data scientists. However, some companies want to get into data science without 

having a relevant team first. You should always thoroughly research any company before going for an interview. Doing that 

often starts by going to the enterprise's website.

What you learn about the company will hopefully reveal that it has a data science team already ??? or at least one devoted to 

analytics. Also, the job description may bring up such a group, such as by saying you'll be leading it or meeting with members

 to work together on projects.

If nothing indicates the company has a data science or analytics team, think at length before accepting the offer. The lack of

 that essential point could mean data science is not a priority for the company, and they might expect you to do too much with 

inadequate or non-existent resources.

If so, you could become so fed up that you end up leaving that job before getting to display your skills and be an asset to the

 company.

6. The Company Is Well-Reviewed---- Part of your vetting process for any company offering you a data science position is to use 

employer review sites that allow current or former workers to give their opinions about the workplaces. A favourable job opportunity

 is likely linked with a company people give positive feedback about.

It's always possible that some disgruntled employees will weigh in about problems not related to the business or position. However, 

if you start to see negative feedback trends ??? such as people saying management doesn't listen to input from employees, the workload

 is too heavy, or the environment is extraordinarily stressful and doesn't allow a work-life balance ??? those are all warning signs.

Keep in mind that you may see some things as positive that others deem as downsides. For example, maybe someone worked as a data scientist

 for the company that offered you a job and said they had to do too much independent work. If you're a highly motivated person who doesn't 

require a lot of direction, that could be a plus for you.

Assess the Matter Thoroughly Before Deciding

Getting a job offer is typically an exciting milestone, and if it's the first one you've received as a data scientist, you could be exceptionally 

eager to take it.

However, it's crucial to be aware of the things on this list and realize that not all job offers are equal in the experiences and 

opportunities they provide.)'''

print (summarize(text))
###Abstraction based summarization...

from gensim.summarization import keywords

text =''''(1. The Job Description Correctly Uses Data Science Terminology---Ideally, a data science job 

description should be specific and include language that indicates a familiarity with data science 

terms and the applications used for the work. If it's vague, you should worry that perhaps the company

 merely wants to hire a data scientist because they have the impression they should. In other words,

 since their competitors are hiring data scientists, they should follow suit. If the job description

 doesn't include relevant data science terms and discuss the applications you'll use, the company may

not have a well-defined purpose and role for the person they hire.

2. A Hands-on Test Is Part of the Hiring Process---It's increasingly common for companies hiring data 

scientists to have them take a data science project test. That means you'll demonstrate your skills in 

a scenario that may mimic the kind of work you'll do for the company. It's also possible the test will

 involve showing your coding skills. Data scientists use various programming languages, including R and

 Python, so you should be prepared to go through a coding assessment, too.

Conversely, if the job description has no mention of a test, that could mean the company isn't serious

 enough about hiring excellent and competent data scientists.

3. There Are Concrete Details Instead of Repetitive Buzzwords---Machine learning, artificial intelligence 

and the Internet of Things are a few examples of exciting emerging technologies that can double as mere 

buzzwords in a context without sufficient substance.

For example, you might come across a job posting for a company that gushes "We want to become the leading 

provider of artificial intelligence for the health care industry" and says "Our smart technology will help 

physicians embrace diagnosis tools powered by the Internet of Things" a few paragraphs later. It sounds great,

 but how will the company accomplish those goals?

If it doesn't go into detail about that, there's a good chance the enterprise is only leaning on those buzzwords

 like crutches and hopes that using them will make their openings sound more cutting-edge and exciting.

The company needs to give clear-cut information about how data science fits into its overall objectives, including

 clarifying the expectations set for any data science professional it hires. Otherwise, you may be looking at a 

hype-filled job description instead of one for a career that helps you grow and provide value to the company.

When businesses have data-centric goals, it's easy for them to conclude it's time to hire data scientists to do

 high-quality work on dedicated tasks. According to one survey, 69 percent of companies were in the process of building

 a people analytics database. A job ad for a good data science opportunity may mention you'd work on a specific project 

like that. Specificity about your duties helps increase a company's credibility.

4. The Company Is Open to You Finishing Current Obligations Before Starting---Maybe you're interviewing for a data science 

job 10 days before earning your degree, or you might work at a place where you haven't provided your two-week notice. In 

instances like those, a decent employer will allow you to tie up loose ends without feeling under pressure. However, if a 

company insists they need to hire you as soon as possible, that could be a red flag for several reasons.

First, the company may be rushing the hiring process, which could mean you get hired for a job that doesn't suit you. Second,

 they might not have thought through the position itself. In that case, you could enter a chaotic environment with little 

direction or leadership.

An arguably worse situation is if the employer tries to make you feel guilty for not starting immediately. They may even say

 something like, "Well, you're really leaving us in a bind, especially since the person you'd replace left so suddenly." Then 

you're left wondering why that person quit. Was it because they disliked the job?

5. The Company Mentions a Data Science Team---Numerous things, including a skills shortage and a growing use of technologies

 that generate data, drive the need to hire data scientists. However, some companies want to get into data science without 

having a relevant team first. You should always thoroughly research any company before going for an interview. Doing that 

often starts by going to the enterprise's website.

What you learn about the company will hopefully reveal that it has a data science team already ??? or at least one devoted to 

analytics. Also, the job description may bring up such a group, such as by saying you'll be leading it or meeting with members

 to work together on projects.

If nothing indicates the company has a data science or analytics team, think at length before accepting the offer. The lack of

 that essential point could mean data science is not a priority for the company, and they might expect you to do too much with 

inadequate or non-existent resources.

If so, you could become so fed up that you end up leaving that job before getting to display your skills and be an asset to the

 company.

6. The Company Is Well-Reviewed---- Part of your vetting process for any company offering you a data science position is to use 

employer review sites that allow current or former workers to give their opinions about the workplaces. A favourable job opportunity

 is likely linked with a company people give positive feedback about.

It's always possible that some disgruntled employees will weigh in about problems not related to the business or position. However, 

if you start to see negative feedback trends ??? such as people saying management doesn't listen to input from employees, the workload

 is too heavy, or the environment is extraordinarily stressful and doesn't allow a work-life balance ??? those are all warning signs.

Keep in mind that you may see some things as positive that others deem as downsides. For example, maybe someone worked as a data scientist

 for the company that offered you a job and said they had to do too much independent work. If you're a highly motivated person who doesn't 

require a lot of direction, that could be a plus for you.

Assess the Matter Thoroughly Before Deciding

Getting a job offer is typically an exciting milestone, and if it's the first one you've received as a data scientist, you could be exceptionally 

eager to take it.

However, it's crucial to be aware of the things on this list and realize that not all job offers are equal in the experiences and 

opportunities they provide.)'''

print(keywords(text))
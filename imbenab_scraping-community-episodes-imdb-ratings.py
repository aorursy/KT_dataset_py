from requests import get
from bs4 import BeautifulSoup
import pandas as pd 
# Initializing the series' that the loop will 
community_episodes = []

# For every season in the series
for sn in range(1,7):
    # Request the server the content of the web page by using get(), and store the server’s response in the variable response
    response = get('https://www.imdb.com/title/tt1439629/episodes?season=' + str(sn))

    # Parse the content of the request with BeautifulSoup
    page_html = BeautifulSoup(response.text, 'html.parser')

    # Select all the episode containers from the season page
    episode_containers = page_html.find_all('div', class_ = 'info')

    # For each episode in each season
    for episodes in episode_containers:
            # Get the info of each episode on the page
            season = sn
            episode_number = episodes.meta['content']
            title = episodes.a['title']
            airdate = episodes.find('div', class_='airdate').text.strip()
            rating = episodes.find('span', class_='ipl-rating-star__rating').text
            total_votes = episodes.find('span', class_='ipl-rating-star__total-votes').text
            desc = episodes.find('div', class_='item_description').text.strip()
            # Compiling the episode info
            episode_data = [season, episode_number, title, airdate, rating, total_votes, desc]

            # Append the episode info to the complete dataset
            community_episodes.append(episode_data)
community_episodes = pd.DataFrame(community_episodes, columns = ['season', 'episode_number', 'title', 'airdate', 'rating', 'total_votes', 'desc'])

community_episodes
def remove_str(votes):
    for r in ((',',''), ('(',''),(')','')):
        votes = votes.replace(*r)
        
    return votes
community_episodes['total_votes'] = community_episodes.total_votes.apply(remove_str).astype(int)

community_episodes.head()
community_episodes['rating'] = community_episodes.rating.astype(float)
community_episodes['airdate'] = pd.to_datetime(community_episodes.airdate)

community_episodes.info()
community_episodes.to_csv('Community_Episodes_IMDb_Ratings.csv',index=False)
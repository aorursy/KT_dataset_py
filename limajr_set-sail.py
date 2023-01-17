!pip install scrapy
import scrapy
starting_point = "http://bible.com/pt/bible/1930/PSA.1.NVT"
import scrapy



class PsalmSpider(scrapy.Spider):

    name = 'psalm_spider'

    starting_point = "http://bible.com/pt/bible/1930/PSA.1.NVT"

    start_urls = [starting_point]



    def parse(self, response):

        CHAPTER_SELECTOR = '.book .chapter'

        chapter = response.css(CHAPTER_SELECTOR).get()

        print(chapter)
!scrapy runspider ./scrapers/scraper001.py --nolog
import scrapy



class PsalmSpider(scrapy.Spider):

    name = 'psalm_spider'

    starting_point = "http://bible.com/pt/bible/1930/PSA.1.NVT"

    start_urls = [starting_point]



    def parse(self, response):

        CHAPTER_SELECTOR = '.book .chapter'

        LABEL_SELECTOR = '.label ::text'

        VERSES_SELECTOR = '.verse .content ::text'

        

        chapter = response.css(CHAPTER_SELECTOR)



        label   = chapter.css(LABEL_SELECTOR).get()

        verses  = chapter.css(VERSES_SELECTOR).getall()

        

        print(label)

        print(verses)
!scrapy runspider ./scrapers/scraper002.py --nolog
import scrapy



class PsalmSpider(scrapy.Spider):

    name = 'psalm_spider'

    starting_point = "http://bible.com/pt/bible/1930/PSA.1.NVT"

    start_urls = [starting_point]



    def parse(self, response):

        CHAPTER_SELECTOR = '.book .chapter'

        LABEL_SELECTOR = '.label ::text'

        VERSES_WRAPPER_SELECTOR = '.q'

        VERSES_CONTENT_SELECTOR = '.content ::text'

        verses = []

        

        chapter = response.css(CHAPTER_SELECTOR)

        label   = chapter.css(LABEL_SELECTOR).get()

        

        for verse_wrapper in chapter.css(VERSES_WRAPPER_SELECTOR):

            verse_contents = verse_wrapper.css(VERSES_CONTENT_SELECTOR).getall()

            verse_text = ''.join(verse_contents)

            verses.append(verse_text)

        

        print("""

        {

            'chapter': %s,

            'verses': %s

        }

        """ % (label, verses))
!scrapy runspider ./scrapers/scraper003.py --nolog
import scrapy



class PsalmSpider(scrapy.Spider):

    name = 'psalm_spider'

    starting_point = "http://bible.com/pt/bible/1930/PSA.145.NVT" # Yeah, I know it. But you know it.

    start_urls = [starting_point]



    def parse(self, response):

        CHAPTER_SELECTOR = '.book .chapter'

        LABEL_SELECTOR = '.label ::text'

        VERSES_WRAPPER_SELECTOR = '.q'

        VERSES_CONTENT_SELECTOR = '.content ::text'

        verses = []

        

        chapter = response.css(CHAPTER_SELECTOR)

        label   = chapter.css(LABEL_SELECTOR).get()

        

        for verse_wrapper in chapter.css(VERSES_WRAPPER_SELECTOR):

            verse_contents = verse_wrapper.css(VERSES_CONTENT_SELECTOR).getall()

            verse_text = ''.join(verse_contents)

            verses.append(verse_text)

        

        print("""

        {

            'chapter': %s,

            'verses': %s

        }

        """ % (label, verses))

        

        NEXT_PAGE_SELECTOR = '.bible-nav-button.nav-right[title^="Salmos"] ::attr(href)'

        next_page = response.css(NEXT_PAGE_SELECTOR).extract_first()

        if next_page:

            yield scrapy.Request(

                response.urljoin(next_page),

                callback=self.parse

            )
!scrapy runspider ./scrapers/scraper004.py --nolog